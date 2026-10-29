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

### Physical page backing

The page table records the bridge rows' backing locations during preparation.
Adjacent locations form one run. A logical selection composes with these runs
before WRITE descriptors are emitted; a local pointer no longer supplies their
physical offsets.

Rows, flat ranges and indexed masks share one logical interval traversal. Both
packet placement and backing composition consume it.

Peers share the compact backing relation once while preparing a WRITE. The
compiler joins physical breaks and registration boundaries with the existing
packet and publication cuts. Each invocation keeps its own cuts when storage
slots have different periods. Source selections may repeat or reverse pages.
A rank with no storage contributes no physical cuts. Contiguous backing needs
only its existing phase report and adds no mapping exchange.

Execution consumes the same immutable DMA descriptors as before. It does not
walk the backing runs, look up bridge rows or copy a payload to change layout.

Preparation on 2026-10-29 checked scattered backing with rows, reversed and
repeated source ranges, unequal slot periods, frame trains and tiled writes.
Every emitted interval retained its selected bytes and stayed within one
registration extent. Contiguous cases produced identical packet tables to the
prior compiler. A short tail of a petabyte-scale logical declaration resolved
from one backing run without page expansion. Shared-library and native builds
completed; no numerical execution ran.

Ordinary collectives carry `mesh_packet_view` from the physical route compiler
through placement. Each view names a byte interval in a common domain.
Reductions, broadcasts and gathers use their shared operand coordinates;
counted transactions retain independent domains with distinct endpoint origins.
Relays retain those coordinates while using their own local storage.

`ncclMeshBufferSlots` declares each input and output's view, stride and slot
count. The native recorder now passes these operands into its collective
statement; it no longer discards them. Native, torch and host declarations use
the header's type. A linear native invocation sequence supplies its declared
invocation count when it resolves its slots. No count is guessed from an address.

Preparation places both internal storage generations and visits each declared
binding period. Endpoints, reduction landings and relays contribute their actual
backing breaks and registration boundaries. A preparation exchange joins these
cuts in common byte coordinates. Fragment endpoints join them too. The regular
page grid stays implicit until descriptor emission. There is no packet-size
exchange during numerical execution.

The earlier whole-element backing splitter and its separate binding exchange
are removed. A backing break inside a row cuts its DMA packets while the
numerical operation keeps the whole row. The same byte compiler handles counted
transactions and ordinary collectives. Compiled packet counts drive channel
prefixes and relay dependencies. Relays no longer require a forwarded range to
start at a nominal page boundary.

A declaration without bindings can describe geometry for demand queries. It
cannot record transfers. A changed binding must name one of the declared slots
before its descriptors are built. These are preparation contracts; GPU execution
consumes the recorded page descriptors and numerical arguments.

WRITE readiness words also resolve through bridge rows during preparation.
Readiness keeps its own slot period, independently of the payload slots.
Submission reads the recorded offset rather than deriving it from a host address.

Numerical landings use `mesh-landing.h`. Each scalar or structured row needs the
packet containing its final byte. Equal page runs and their short final packet
remain one arithmetic record; kernels do not walk packet lists. Structured
merges can use their early rows before the whole message arrives. The former
square-root packet threshold is removed. First combines use this same landing
operation with the original input as an explicit operand. The separate
`metal_combine3` interface, kernels and preceding whole-message wait are removed.
The backend records that input's GPU address during preparation and retains its
resource with the recorded work; execution has no extra mapping lookup.

Structured merges share one numerical body for original-input and in-place
bindings. Preparation selects the known alias relationship. The in-place form
keeps its existing no-work case when the other partial is empty; the separate
output form writes the retained input in that case. There is no runtime alias
test. Both consume the same compiled packet-readiness relation.

Preparation on 2026-10-29 checked emitted scalar and structured landing records
with either input placement and with packet breaks inside numerical rows. Input
addresses, retained resources and readiness constants agreed with the declared
bindings. The Metal kernels compiled. No numerical dispatch ran, so this gives
no new latency or useful-bandwidth result.

Preparation on 2026-10-29 exercised the real route compiler and packet-cut
preparation with three-rank relays, uneven all-to-all origins, scattered backing,
different slot periods and both storage generations. Every physical pair had
the same byte packets. Each packet stayed within its backing run and registration
extent; numerical readiness matched the final byte of each complete row.
The configuration exchange used in-process peers and synthetic backing metadata;
no payload was allocated or sent. Shared libraries, header-derived Python
bindings, the torch extension and native callers built. These checks do not
provide a new throughput or latency measurement.

Tensor allocation now takes free bridge pages independently of their arena
positions. Preparation maps adjacent backing runs into a fixed local virtual
window in logical page order. CPU, Metal and CoreML operands see one interval
over those same pages. No data is copied, and execution does not remap pages or
walk the backing relation. Preparation shares the bridge's existing mapping
with `mach_vm_remap` and its copy flag false; it does not allocate a second
payload pool or reopen the backing object. Apple's implementation extracts the
source mapping before replacing the target interval under the map lock
([XNU `vm_map_remap`](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/vm/vm_map.c)).

Local ownership lookup uses that window's row indices. Registration, page-table
creation and transport preparation no longer require the local view to lie at
the bridge's own address. Control words and descriptor bindings resolve to
bridge offsets through the shared page-table operation during preparation.

The bridge currently reads each descriptor table as one linear region
interval. Those metadata allocations explicitly request linear backing through
the same allocator. Tensor allocations have no such requirement. Both use the
same local view and release rules; neither retries through a copying allocator.
The shared bridge layout and recorded execution descriptors are unchanged.

Preparation on 2026-10-29 used a small fragmented shared-memory region with
different page-block widths. Large logical operands occupied separated holes;
CPU metadata writes were visible through the corresponding bridge pages.
Metal buffers and CoreML arrays retained the same addresses. Linear descriptor
tables, partial-allocation cleanup and released-slot reuse also prepared under
address and undefined-behavior sanitizers. No numerical commands or model
predictions ran. A CoreML array over those pages does not establish that an
ANE provider honors an output backing during prediction.

Consecutive page-table row slots and linear descriptor-table backing remain
allocation requirements. The tensor's payload no longer needs consecutive
arena pages, so fragmentation of that arena alone no longer imposes its former
whole-tensor limit. Distributed provider cuts and ANE output placement still
need the integration described below.

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

Strides, axis maps and joint maps describe the same kind of view. A joint
table becomes one index factor; it has no separate storage or emission path. Slice,
selection, permutation, reshape and gather compose those relations before
execution. Results retain no parent descriptors. Component views subdivide
typed elements for byte-preserving copies. Known maps simplify to arithmetic
and compact index expressions where possible.

These maps are compiler metadata. Their constancy says nothing about values
in the tensor, including values used as indices. Both static and dynamic
indexing need declared dependencies. A scale, rotation or reduction remains
numerical work; a layout change cannot stand in for it.

### Compact reshapes and aliases

`shape` describes logical axes. `indexShape`, strides and maps describe the
factors of its storage relation. `coordinates` maps a logical flat index to
those factor digits. A reshape changes logical axes without changing the
relation. A cut can select fewer values than the factor domain contains.
Neither operation needs a parent descriptor. Numerical indexing and dependency
records consume the same factors and coordinates.

Adjacent affine axes form linear runs, including
negative and zero strides. New axes split those runs. An axis map stays one
map when only singleton axes change, and splits into smaller maps when its
values separate across the new axes. The surrounding rows add no index words.
These are normal forms of `MetalTensorLayout`; callers use the same interface.
Component-byte views scale the factors and append a component axis. Removing
that axis restores the typed view without repeating each scalar's map entries.
Axis removal uses the shared reshape operation; logical axis numbers do not
index address factors after a nonseparable reshape.

Rebasing a shape alias applies that reshape to its assigned pages. Affine
subviews use the declared coordinate basis and its bounds to compose their
new origin and strides. Slices, permutations and broadcasts therefore retain
affine bindings when the placement allows them. The shared coordinate inverse
also serves the general indexed composition. No relation object or coordinate
search is introduced into numerical execution.

Preparation on 2026-10-28 described 2,061,584,302,080 logical elements with six
axis-map entries, then factored the same map into strides alone. Large sliced,
permuted, reversed and broadcast aliases also retained strides. Sampled index
relations agreed, and the emitted index declarations compiled as Metal source.
These descriptors allocated no tensor payload. Complete mapped reduction and
gradient graphs prepared; native rank and Goal A built. No numerical commands
were submitted and no throughput or latency improvement is claimed here.

The later 2026-10-28 pass retained six nonseparable map entries for that same
2,061,584,302,080-element domain after changing its logical axes. Serialization,
component-byte round trips and whole-view alias placement preserved the source
index expression. Python, Swift and LiteRT declarations carry the same index
factors. The LiteRT importer compiled the same compact declaration. Shared
library, frontend and native builds completed; no numerical commands ran.

Selections and permutations now separate independent domains at shared
logical/index boundaries. A cut composes the domain containing its axis.
A permutation moves independent domains intact and combines domains only when
their coupled axes interleave. Unchanged rows do not repeat a small map.
The result is the same flat layout; these domains exist only in preparation.
Moving singleton axes remains a reshape with the original index expression.

Rectangular aliases of affine declarations use that same composition when
their producer receives mapped pages. The coordinate relation supplies slices,
reversals, permutations and broadcast axes. It does not build an inverse table
over the whole producer merely because its assigned storage is mapped.

Preparation on 2026-10-28 kept six map entries after cutting or transposing a
68,719,476,736-row view, and four after selecting part of its coupled domain.
An interleaved permutation composed 30 entries while preserving the row factor.
Two coupled groups likewise composed 60 entries while moving a billion-row
factor intact. Alias placement retained compact cuts, transposes and broadcasts.
Preparation covered negative and zero strides, repeated selections, singleton
and empty axes, serialization and ordered/dependency spans. Shared libraries and
Metal declarations compiled without numerical submission.

An independent index boundary can fall inside an unchanged logical axis after
a reshape. Composition now splits that logical axis at the boundary, performs
the selected cut in its own domain, then restores the declared logical shape.
It does not repeat the selected map across the unchanged extent.

Linear cuts also resolve against the index factors before building a table.
When the selected interval is a rectangular range of those factors, preparation
cuts their digits directly. A stride which divides a factor fixes its residue;
a negative step reverses the selected digits. Repeating one value uses a zero
stride. The result uses the same layout type, serialization and source emitter.
No additional index relation is read during execution.

Preparation on 2026-10-28 retained six entries for aligned cuts, full reversals
and strided selections of a 2,061,584,302,080-element flat view. Cuts through a
logical axis with an independent row factor retained four entries. Smaller
compositions preserved exact indices, ordered transfers and dependency spans;
the large declarations compiled as Metal source. No numerical work ran.

Nonrectangular cuts and interleaved permutations now substitute integer
coordinates into the existing factors. The preparation terms contain constants,
the logical index, addition, integer scaling, floor division and remainder.
Normalization removes terms and folds fixed residues before source emission.
Shaders receive ordinary integer expressions and the existing map entries;
there is no expression interpreter, parent view or extra lookup per transform.
Computed page indices remain ordinary tensor values, not these compiler terms.

Byte components, alias placement, saved FX declarations and native LiteRT
`index_coordinates` carry this relation. A byte view scales the same factors;
removing its component axis restores the typed view. A nonrectangular alias of
a dense affine declaration uses its coordinate inverse. An alias retaining its
producer's factor basis composes directly with the assigned placement. Ordered
transfers follow prepared affine runs of the relation. Dependency spans use the
same runs and the shared range union, including repeated and reversed reads.
Preparation checks alignment and adjacent lanes algebraically; it emits vector
accesses when that relation holds and adds no runtime alignment guard.
When a selection leaves only an affine part of a map, normalization restores
an ordinary strided binding. A strided consumer can then use it directly.

Preparation on 2026-10-29 retained six map words for nonrectangular cuts and
interleaved permutations of a 2,061,584,302,080-element domain. Their coordinate
JSON occupied at most 148 bytes. Smaller compositions preserved ordered
transfers, exact byte-access unions, byte-view round trips and saved declarations.
The emitted Metal source compiled. These are layout-preparation results, not
numerical execution or throughput measurements.

Explicit selections now compose each factor with the selected positions during
preparation. Common coordinate terms stay arithmetic. An unchanged map stays
one map. Other terms separate their repeated residue from their linear advance;
only the distinct residue and selected positions become a table. Each table
contains final storage indices, so execution adds no nested map lookup. Native
and FX consumers use the same result, serialization and source emitter.

Preparation on 2026-10-29 selected repeated, out-of-order columns across the
factor boundaries of a 606,348,324,135-element view. The result held 300 map
entries and 145 bytes of coordinate expressions, the same as its smaller
875-element counterpart. Exact ordered transfers, dependency spans, byte views
and serialization agreed on smaller declarations. Metal source compiled without
allocating tensor payload or submitting numerical work.

Alias placement now composes independent mapped factors through the same
selection operation as affine factors. The declaration supplies the source
relation and assigned pages. Preparation resolves the selected coordinates,
then applies their cuts, reordering and repetition to the assigned relation.
An affine outer extent does not repeat the inner map or acquire a table of its
own. This also covers an affine cut through an irregular source axis.

The general inverse no longer builds a dictionary over every producer element.
It bounds intervals of the declared coordinate expressions and resolves affine
runs directly. Range summaries cover only the supplied maps. Repeated address
patterns restrict the search to their last period, preserving the existing
representative for aliased values. Repeated selections likewise emit one period
of output indices. All of this is preparation work; execution consumes the
resulting flat layout. Poorly separated relations can still need many interval
searches.

Preparation on 2026-10-29 rebased an 824,633,720,688-element mapped cut with four
map entries, including reordered columns, repeated columns and a reversed
axis. A 687,194,767,360-element affine cut through a mapped source retained no
map entries. A cut through two mapped axes covered 4,123,168,603,740 elements
with seven map entries. Sparse aliases into a 2,061,584,302,080-element producer
resolved without enumerating that producer; a trillion-element repeated alias
needed fifteen output entries. Smaller declarations preserved exact ordered
transfers, dependency spans, typed views and serialization. Their Metal index
source compiled. No tensor payload was allocated or numerical work submitted.

A coupled selection or alias with no compact composition can still materialize
its selected domain. The distributed ownership gap under exact partition
boundaries also remains unfinished. An irreducible table supplied by a caller
occupies its own domain; reshaping, slicing or permuting it no longer duplicates
that table across surrounding dimensions.

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
expose its partial sum before a head can span nodes. The vector and tile
attention producers now mask channel tails and bound their working storage,
as described below. The model loader's 256/512-only head filter is removed.

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

### Distributed reduction declarations

`algebra.py` now supplies softmax, log-softmax, their gradients and vector
norms to both FX capture and the ragged tensor frontend. The latter consumes
the same decomposition table before sharding propagation. Its separate
linear-gradient adapter is removed. These are graph rewrites during
preparation; replay receives their numerical records and page operands.

Softmax exposes a maximum and a sum per retained row. Its gradients expose
row sums. A vector norm exposes sum, maximum or minimum after transforming
the local values, then applies its result operation. These ordinary reduction
operators already declare collectives. They carry summaries across reduced
mesh dimensions while the input values remain on their assigned nodes. Mean
continues to divide by the logical count, rather than the local shard size.
The algebra does not insert a contiguous copy or require the full row on one
node. Empty softmax domains resolve during declaration without a reduction.

The FX numerical emitter also takes its reduction launch limit from
`ROW_THREADS` and the device capacity, and its SIMD width from the manifest.
It no longer inherits those constants from upstream Metal code generation.

Wide reductions with few output rows also split into partial reductions during
IR lowering. `REDUCTION_TILE_ELEMENTS` controls the initial work per part;
`REDUCTION_GROUPS_PER_CORE` and the discovered GPU core count set the target
parallelism. Both choices are compiler operands with empirical parameter
domains. They do not cap a tensor's width. Small rows remain one reduction;
larger rows produce page-backed summaries and a merge. Half and bfloat
intermediates retain the IR's float accumulation. The same prepared recorder
consumes both stages and their dependencies, without a second executor.

Preparation on 2026-10-28 compiled complete forward/backward graphs with
mapped half, bfloat and float inputs, including odd widths and 1,048,576
channels. Different declared reduction launch limits reached source emission.
FakeTensor tracing with metadata-only process groups described uneven shares
over four ranks and over two mesh dimensions. The recorded collectives carry
row summaries, with no input all-gather in those graphs. These groups did not
communicate, and neither these declarations nor the native numerical graphs
were executed. This is structural evidence, not a cluster or performance run.

Further preparation split two mapped rows of 1,048,577 values into 160
threadgroups on the discovered 40-core GPU, followed by summary merges.
Transposed views used the same decomposition. Changing the declared work
targets changed the emitted splits; short rows kept one stage, and empty
outputs emitted no numerical commands. Forward/backward softmax and norm
graphs also compiled with this split policy. No numerical replay was run.

The multidimensional declaration still contains a separate all-reduce for
each reduced mesh dimension. Softmax still declares maximum and sum separately;
folding them into the existing scaled-partial merge is unfinished. Other
whole-dimension rules in `partition.py` are also still present.
Native fused row norms still need their distributed factor operands connected
through the complete graph. These are remaining lowering and fusion tasks.
Native attention now carries logical channel extents through both producer
forms; the model loader no longer restricts heads to widths 256 and 512.

### Centered moments across owners

The shared algebra now expands functional variance, standard deviation and
their paired mean forms into a mean and a centered-square sum. Both reductions
use the declared global count, including the caller's correction. Unequal and
empty local shares do not change that denominator. Complex inputs contribute
the squared real and imaginary parts; low-precision inputs use the computation
type until the result conversion. These are ordinary graph operations over
the existing page operands, with no new numerical executor.

Preparation described a normalization and gradient graph at width 1,048,577
on all four ranks, including an empty share. Its crossings contained row
summaries rather than input gathers. Complete mapped native graphs compiled
at widths 37, 32,769 and 1,048,577 with half, bfloat and float inputs. Scalar,
empty and multi-axis metadata also lowered. No numerical commands were run.

That preparation exposed a shared emitter error: a reduction of a broadcast
value chose its accumulator before a later fused load opened the reduction
loop. The page emitter now establishes every reduction domain first, so its
accumulator covers all loop iterations and remains in scope afterward.

The small distributed backward graphs still expose whole-input gathers chosen
by upstream placement. At widths 3 and 37, it keeps a broadcast gradient partial
and gathers the sharded value it multiplies, instead of reducing the small
gradient first. Repeated mean-summary reductions also remain in the declared
graph. Those placement and common-work issues are unfinished; the wide graph
does not establish their absence for other shapes or complete distributed
moment support for mutable output forms.

### Exact partition boundaries

`allocate.partition_bounds` now supplies the same integer prefix boundaries
to tensor partitioning, migration writes and context position maps. A width
need not be a multiple of the capacity units. Adjacent cuts cover the whole
extent; divisible widths retain their existing cuts. Empty shares carry empty
page operands and reduction identities. Context head/tail layouts divide each
half by this relation, including odd lengths; their maps contain no padding.

A local size cannot recover a rounded global extent. Ragged `from_local`
therefore consumes the declared shape and stride. Gradients keep that metadata
from their source specification. Row-parallel linear inputs take the declared
input width from the module. The TP example keeps the distributed operand at
module and residual boundaries instead of discarding and reconstructing it.
Context consumers likewise require the declared sequence length.

Shape-only views compare the actual affine regions assigned to each rank
before choosing their output shard axis. This bypasses upstream's requirement
that a flattened axis divide the number of ranks. Compatible cuts stay local.
The comparison retains each shard's logical element count: broadcast axes
repeat values even though they add no addresses. It must not equate a six-value
local view with a proposed five-value shape merely because both repeat one page
value. That case previously selected the wrong shape and failed preparation.
An incompatible cut still declares a gather: the current `RaggedShard` cannot
describe every mapped output selection. Replacing that gather with the common
page-selection relation remains unfinished; this change does not claim that
all reshapes retain distributed ownership.

Preparation on 2026-10-28 traced all four ranks at widths 3, 37 and 1,048,577,
including empty local reductions, softmax, vector norms and their gradients.
The crossings carried row summaries. A two-linear TP graph at widths
5 → 37 → 13 retained its residual and gradients on each rank. Context unshard
declarations covered zero, odd and short sequences. Migration descriptors
used the same source and destination cuts, including zero-share destinations.
Compatible views emitted no gather; an incompatible flatten exposed the
remaining gather in its graph. These were metadata declarations and compiled
maps, with no numerical replay or cluster performance measurement.
The ordinary TP example also traced its full forward/backward graph on rank
zero with the distributed operands retained across module boundaries.

### Partial-result merge capacity

The shared `mesh_merge` no longer limits a row to 64 contributors or requires
its channel slices to fill complete SIMD groups. It reads weights in bounded
batches, using the register budget in `metal.def`. Each batch continues the
same accumulation over the row's parts. A final partial SIMD group uses its
active lanes; channel groups may cross SIMD boundaries.

The fused native producer also bounds its shared weight batch. Its declaration
passes the scratch capacity to `mesh_merge`; the contributor count no longer
serves as a scratch index. The first SIMD group fills one batch, all groups
consume it, and a barrier orders the next overwrite of those same slots. A
single-batch merge retains its existing publication barrier and adds no reuse
barrier. The normalizer and value sums continue across batches. This removes
the overflow when a producer has more contributors than its page-tile scratch.

Native preparation now sizes partial-result pages from all of a network's
declared row classes and layers. `AttentionStorage` is a tuple of prepared
buffer bindings passed to command construction. It does not own execution.
Successive calls use the prepared maxima, total, value and count pages; their
access ranges still come from each call. The old global arrays and their
`PARTIAL_HEADS` capacity are removed. An explicit part count is no longer
rejected because those arrays were too small. The configured default part
selection remains a planning input, separate from storage capacity.

Logical width and launch width are separate compilation operands. Native and
LiteRT preparation split wide outputs into channel tiles. Each tile merges
the same partial-result records into its own output channels. A row can span
many threadgroups without adding a runtime allocator, choosing another merge
body or copying the row to change its layout. Its last tile masks absent
channels. Native row offsets and LiteRT slice offsets use full address width.

The partial record still contains two scalars and exactly the logical number
of values. Provider bindings may include physical tail padding; that padding
does not enter the partial records. Both adapters call the shared merge. The
importer no longer requires widths divisible by 128.

LiteRT partial rows now carry explicit storage operands from
`configs/litert-reference-graph.json`: physical axis order, scalar type, scalar
byte width and lanes per vector, separately for values, normalizers and merged
results. Preparation binds symbolic extents and builds `MetalTensorLayout`.
That same layout emits the shader index expression and the recorder's access
ranges through `MetalDispatchAccesses`. A holder writes directly into its
declared part of the shared record. Non-unit leading axes and batch extents on
several axes no longer use the old flattened slice formula. Output padding is
zeroed by the existing merge command. Zero-channel exports retain their two
summary scalars; their empty merged value emits no command.

Native stage extents use integers, named configuration operands and products
of those operands. The old `per_sequence` and `batched` flags are gone. Their
values, layouts and launch geometry are resolved before recording. An explicit
threadgroup size is not silently reduced after compiling its tile arithmetic.
Execution gains no descriptor parser, layout copy or launch selector.

Preparation on 2026-10-28 compiled native merges from width 1 through 65,537,
contributor counts through 1,025, and partial SIMD launches. The ordinary
merge shaders use zero static threadgroup memory. The actual LiteRT importer
prepared tiled row export and merge stages, including odd widths and width
65,537; their shaders compiled. Existing vector, tile and int8 attention
producers also compiled. Native rank and Goal A built. No numerical execution
or throughput/latency result is established by this preparation.

The provider-layout extension also prepared non-unit leading axes, symbolic
batch extents on one and several axes, zero channels, odd widths and width
65,537. The actual importer and native shader compiler consumed independently
declared half, bfloat and float layouts with two-, four- and eight-lane storage.
Sampled indices matched the provider's physical ordering. This establishes the
adapter's lowering and source compilation, not support for those alternate
types in the packaged delegate or numerical equivalence of a complete model.

The fused-batch extension compiled shared capacities of 17 and 64 with up to
1,025 contributors, including partial SIMD groups and an empty set. Actual
native vector producers specialized for 64, 65 and 1,025 contributors kept the
same static threadgroup memory at each fixed head/launch shape: 1,360 bytes at
width 256 with one SIMD group per page, and 2,672 bytes at width 512 with two.
Native rank and Goal A built after the page-sizing change. These are preparation
results only; no numerical work or timing/throughput measurement was submitted.

This removes limits from the merge of already-produced partial results. The
producer lowering is described next. LiteRT's native selection adapters now
consume the shared views described below. The delegate's internal tensor views
remain a separate integration requirement. Placing
the partial numerical work across nodes remains part of the graph's reduction
lowering, rather than a property of this merge.

### LiteRT selection views

Draw, retained logits and the final token selection now consume
`MetalTensorLayout`. The importer passes each graph tensor's full shape and the
provider's configured scalar type, byte width, lanes and physical axis order.
`selection_storage` and `partial_row_storage` declare `orders` by tensor rank.
A missing order has no inferred replacement. Leading axes and symbolic batch
extents survive into the same source-index and dependency-range lowering.
The old `litert_slice` and `litert_at` formulas are removed. Their other users,
prefill-mask stores and activation-row selection, also consume declared views.
The mask word layout is a configuration operand. Activation selection receives
the source shape, type and lane width, including its own padded row extent.

Partial-row and selection adapters share `provider_view`. Native stage
preparation also passes explicit axis maps and joint indices to the shared
layout algebra. These are declared storage relations; computed indices remain
runtime page contents. No descriptor walk or layout copy is added to execution.

The importer declares retained logits using the draw's same input operands.
`native_call` prepares both through one implementation, and `record_native`
records both. The former host reconstruction of logits names, row counts,
types and a separate pipeline is removed. Retained record fields use the
shared selection function's write callback and their declared output view.
The old pointer-pair call no longer matched that shared function.

Preparation on 2026-10-28 compiled selection declarations with two, three and
four axes; odd and zero vocabulary widths; half, bfloat and float storage;
two-, four- and eight-lane provider layouts; and symbolic batch extents on
multiple axes. Explicit axis and joint maps used the same native preparation.
Partial-row adapters also prepared after adopting the shared provider helper.
The mask and activation-selector shaders compiled with odd widths and a
non-unit leading axis. Native libraries and rank built. No tensor payload or numerical commands were
submitted by those preparation programs. This does not establish alternate
storage support in the packaged delegate, complete-model numerics or measured
latency/throughput.

### Bounded attention tile storage

`mesh_attention_tile` and `mesh_tile` now carry logical width through query,
key, value, accumulator and output indexing. Matrix fragments mask the final
channels. Partial-result rows contain exactly their logical values, even when
their width is odd. A shared-prefix tail shifts its output by scalar elements;
it no longer rounds that offset down to whole four-value vectors.

The query cache has a byte budget in `metal.def`, checked against the device's
declared threadgroup capacity. Within that budget, the query matrix is staged
once and its final matrix columns are zeroed. Beyond it, matrix fragments read
their query values directly from the input pages. This is a source
specialization, with no runtime storage choice or descriptor walk.

The accumulator register budget is also an operand in `metal.def`. A row which
fits retains its accumulators across key chunks. Wider rows use bounded column
batches and keep their running float sums in the already assigned partial-result
pages. The score calculation and online softmax run once per key chunk. Each
column batch reads those same probabilities and updates its own values; it does
not repeat the score calculation. No extra dispatch or payload allocation is
introduced.

Matrix accumulators now write their assigned output positions directly. The
former full-row shared-memory staging, barrier and copy to partial-result pages
are removed. Native tile publication and its fused final merge also cover
channel tiles, including odd tails. Widths 256 and 512 retain their existing
number of concurrent query rows in that merge.

Preparation on 2026-10-28 compiled the actual native tile and shared-prefix
producers from width 1 through 1,048,577, including odd widths, received queries
and fresh KV operands. With the declared budgets, widths above 512 used 3,376
bytes of static threadgroup memory in the tile producer and 3,360 in the
shared-prefix producer. The width-256/512 vector producers and bounded merges
also compiled. Native rank and Goal A built. These were source compilation and
builds only; numerical equivalence and whole-function latency/throughput remain
unmeasured.

### Bounded vector channels

The vector producer also separates logical width from its physical launch.
Preparation derives channel tiles and position groups which divide that launch
and its KV pages. A final tile masks channels beyond the declared width. Query
cuts use a ceiling division, so the inner product includes the final channels.
Query registers have a per-lane budget in `metal.def`; larger queries read
fragments from their input pages.

The shared `mesh_attention` computes each key chunk's scores and softmax once.
All channel tiles consume those probabilities. A single tile keeps its sums in
registers. Wider rows keep sums in their assigned partial-result pages between
chunks. The existing completion word covers the whole row; channel batches do
not create new completion counters or queue submissions.

The old `native.metal` int8-window and projection stages were orphaned: commit
`19793ca2` in metal-microbench removed both planners on 2026-10-14, and no current
importer or stage named either kernel. Their remaining source was removed on
2026-10-28. The active LiteRT graph uses its imported model operations and the
shared partial-row/merge adapters. There is no active one-batch native int8
operand to extend or retain as an alternative executor.

Both vector and tile producers now preserve logical row strides and mask tails
on reads and writes. The ordinary model loader's 256/512 filter is removed.
Preparation compiled vector widths 1 through 1,048,577 with both configured
launch splits, one and multiple contributors, fresh KV and received queries.
Wide rows used 320/336 bytes of shared storage at 128 threads and 608/624 bytes
at 256 threads, depending on whether the fused contributor merge was present.
The previous 256/512 specializations retained their shared storage sizes.
Native rank and Goal A built. No numerical commands or performance measurements
were run.

Placing one head's inner-product reduction across several nodes remains
separate graph-lowering work. Removing the local width filter does not finish
that distributed lowering or establish numerical results for a complete model.

### Host storage contracts

CPU contraction workspace is declared during preparation. The compiler asks BNNS
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

The native ANE allocator now uses its declared allocation grain independently
of the model build block. It can assign intervals within one block, and ranges
whose ends are both off that grid. Preparation clips each block to the exact
selected columns and retains the row tails supported by `neural_engine_model`.
The model manifest, preparation identity and native binder carry those same
ranges and feature requirements. The configured block still bounds one model
conversion's memory; changing an assignment does not enlarge that bound.

Changes to these constant-weight shares use the existing asynchronous model
preparation path before the new assignment becomes executable. The program key
includes the exact model ranges, so a ragged range change cannot select an older
manifest. Only the assigned blocks are built. Compiled shares count toward the
ordinary program cache budget; the former permanent-block exemption is removed.
Backend assignments now carry lists of intervals. The existing allocator's
configured side yields one interval per engine; explicit disjoint assignments
pass through the same planner record, preparation identity and model manifest.
The builder clips every assigned interval against its build blocks. Native FFN
ownership retains an ordered column selection, and the GPU consumes one composed
view of its gate, up and down matrices. The binder selects the ANE model blocks
for that same assignment. Rate accounting sums the intervals' actual work.
Goal A's independent 32-neuron alignment restriction has been removed.

Preparation on 2026-10-29 carried disjoint assignments through member identities,
clipped model blocks and the native manifest decoder. Native rank and Goal A
built. No model prediction or numerical command ran. The provider output-page
contract and placement control described above remain open.

Preparation on 2026-10-29 carried off-grid intervals through allocation and
member identity, including both start and end assignments. The shared model
builder converted 19 rows as 7/7/5 tiles and 13 channels as 5/5/3 blocks. Core ML
compiled that graph and reported both CPU and ANE support for its operations,
but preferred CPU placement for this small shape. The ANE-only configuration
therefore still declines that provider placement. This preference is not an
alignment requirement and has not been promoted into one. Provider placement
control remains open alongside the output-backing contract. No prediction or
numerical dispatch was submitted; temporary compilation artifacts were removed.

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

### LiteRT constant cuts

`configs/litert-storage.json` in metal-microbench declares scalar widths, the
storage word type and the partition grain. `tools/litert_storage.py` consumes
those operands during graph preparation. Its cut copies encoded bit ranges on
any logical axis. A partial byte stays part of the selected value; the importer
does not round a column boundary down to a byte. Byte-aligned cuts use ordinary
array slices. Unaligned cuts compose storage words without unpacking the entire
constant or producing an element-index table.

The same `select_constant` takes the tensor's sparse storage relation. Traversal
order, dense dimensions, CSR segments and indices, and block dimensions stay
with the encoded values throughout a cut. For a cut within a block, the compiler
uses the greatest block width which divides the old width and both cut
boundaries. It retains the other block dimensions. It rebuilds sparse relations
over stored runs, preserving trailing dense values as runs and joining adjacent
source ranges before copying. It neither expands missing values nor inserts a
runtime conversion. The storage declaration supplies the permitted index-vector
types; their required width follows from the emitted metadata. These fields
follow the [LiteRT schema](https://github.com/google-ai-edge/LiteRT/blob/main/tflite/converter/schema/schema.fbs).

Outer and decomposition graphs use the same `select_graph` operation for constant
data, logical shape and quantization fields. A changed constant gets its own
buffer, and identical cuts share that buffer. The sharing key includes the sparse
relation, so aliases with different traversal orders remain distinct. Other aliases retain the original
constant. Graph extraction and slicing share the same buffer collector, so
unreferenced original constants do not remain in the exported model.
The former higher-axis and decomposition-constant rejection paths are
removed. The planner consumes the declared grain directly; it no longer adds
an implicit 64-column alignment. These are preparation writes to the exported
model, not copies added to numerical execution.

Selections use ordered `(first, width, step, count)` records. One record can
describe an interval, a repeated stripe, a reversal or repeated values. Several
selected axes use this same operation for encoded values, logical shape and
quantization fields. An identity selection retains the original buffer. Sparse
selection finds the selected stored coordinates without enumerating absent
values or empty repetitions. Preparation on 2026-10-29 retained two stored rows
in a selection of half a billion logical rows, and preserved scalar bits,
quantization fields and sparse metadata in ordered and repeated selections.

`partition_axes` and `slice_axis` carry declared axes through projections,
pointwise uses and composite graphs. FFN partitioning starts from the down
projection's input axis and uses those same operations as head partitioning.
The old search for every tensor dimension equal to the FFN width is removed.
Equal batch, neuron and output extents do not make them one partition axis.
Broadcast operands carry the relation on their own rank, and projected biases
follow the output-channel relation.

The partition mark `(axis, unit)` selects a digit in each repeated group on that
axis. Reshape carries the digit by its flat span: a channel selection in
`[batch, heads, channels]` becomes stripes in `[batch, heads * channels]`.
Shape constants retain the whole selected dimension, including its repeats.
The constant selector consumes one compact record for those stripes.

Preparation on 2026-10-28 preserved dense INT2/INT4 tails and fixed-width scalar
bits across axes, odd extents and empty domains. Cut graphs serialized with
uncut aliases intact; a composite's internal constant was sliced through the
same operation. A square FFN retained its batch and final output extents while
cutting its neuron axis and a broadcast scale. No model execution or performance
measurement ran. The later sparse preparation preserved encoded values and
stored positions across reordered traversal, multiple CSR axes, empty cuts and
both axes of a block. Its billion-row declaration retained one nine-value run
without constructing coordinates for absent values. FlatBuffer serialization
retained the revised metadata, and differently interpreted aliases received
different cuts. No numerical commands ran. Partitioning a head's contracted
channel axis and selecting providers for their accepted sparse formats remain
separate graph-lowering work.

### Partial contraction boundaries

LiteRT section preparation now finds the output contraction in the result's
dependency cone. It no longer assumes that the final two operators are a
projection and a dequantization. `contraction_parts` takes the chosen operation
and accumulator type and produces two ordinary graphs through `mesh_graph`.
The first emits additive partials. The second adds the original bias, applies
the original fused activation and restores the result's original encoding after
the collective sum. Its later consumers retain their operators and dependencies.
An empty finish is an input/output alias and emits no numerical operation.
This preserves the declared rounding boundary; it does not promise bitwise
agreement between different floating-point reduction orders.

The old `rounded` helper is removed. It selected a quantizer from elsewhere in
the section and reconstructed the finish from operator positions. Partitioning
now cuts the partial graph through the existing axis and constant slicers. The
finish stays after the sum, including for an odd sparse contraction cut.

`engine.crossing_value` declares the scalar type and width for ordinary value
crossings. The native configuration declares float16 and the LiteRT reference
configuration declares float32. The planner carries this operand into the
program, byte counts, gather counts and crossing types. The importer consumes
the point's type for its partial result. The former float16 transport declaration
over float32 LiteRT partials is removed. Auxiliary graphs use the same collective
record builder, including its dtype and recipient fields.

Preparation on 2026-10-29 serialized a sparse, quantized odd cut with bias,
activation and distinct input/output quantizers. It also retained an identity
finish without an added operation. The reference model's decode attention and
MLP sections prepared head and odd-column cuts through these same functions.
Configured float32, float16 and bfloat16 planner records retained their scalar
widths and transfer extents. No numerical execution or performance measurement
ran. This completes the existing section-output contraction boundary; it does
not yet lower a contraction inside a head across its channel axis or establish
which providers accept each sparse format.

`partition_stages` now also exposes internal contracted values. Batched matrix
products use their transpose flags and aligned batch axes to carry the cut in
either direction. The shared partial/finish split retains each consumer after
the sum, including quantization and nonlinear operations. Fully connected
products propagate both their input and output relations instead of ignoring
one when another is already marked.

The importer expands only composite decompositions which contain such a sum.
Returned aliases remain tensor identities through extraction and append. Stage
metadata carries their names to the existing page binder; no reshape or copy
command is added. The ordinary member builder emits internal crossings with the
configured accumulator type and symbolic transfer extents. Preparation resolves
those extents and aliases before recording. Execution consumes the same recorded
collectives as section-output sums.

Preparation on 2026-10-29 serialized uneven internal and final sums through that
member builder, including composite graphs and repeated returned aliases. The
nonlinear consumer stayed after the internal sum. Transposes and broadcast batch
axes retained their shapes, the reference-model cuts still prepared, and the
LiteRT library built. No numerical commands ran. Reshape relations spanning
several target axes and simultaneous graph cuts on several axes of one tensor
remain open. The planner also still declares whole-head shares.

Static review on 2026-10-29 found a separate external binding gap. The model
export retains a cut tensor's shape, but its ordinary provider binding uses a
name and offset zero. A consumer selecting columns from an earlier full result
therefore loses the relation between its local columns and that result's pages.
Serialization and graph-order checks do not establish correct numerical binding.
The original-to-selected relation must reach the shared layout placement and
provider requirement lowering. A required contiguous view must use direct
placement where possible, or the shared prepared layout-copy operation where
the provider's fixed layout requires it. Numerical execution remains deferred
until that relation is carried through the boundary.

### Encoded numerical operands

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

Native compressed matrices now take explicit selections on both axes through
`MetalViewBlockGather`. Interval cuts feed that same operation. Product lists
and FFN views carry the selected coordinates into each constituent matrix;
they no longer reduce an input selection to one interval. The encoded numerical
blocks stay in their original pages. Preparation selects their metadata and
declares the retained lanes and dependencies.

Both the int8 rotation and its folded scalar weights retain `TensorView` through
column selection. The folded product binds that view's byte components directly,
including reordered and repeated columns. Converting it to a buffer plus row
pitch no longer discards the mapping. There is no layout-only weight copy.

A selected Hadamard input carries a relation from each retained basis position
to its source values. Flat offsets and indices store this relation during
preparation. Repeated selections sum their contributions before the butterfly;
absent positions contribute zero. The fused schedule and shared factor kernel
consume the same fixed-width source groups, emitted by the compiler. FX exposes
that source width in the stage declaration and transposes the same relation in
its backward graph. No runtime object traverses the preparation relation.
The current emitter pads those groups to the largest source count in a domain;
finer grouping for strongly uneven repeat counts remains lowering work.

Preparation on 2026-10-29 built rank and Goal A, composed reordered, repeated and
empty input selections, and compiled the fused and factored shaders with both
ordinary and repeated sources. FX fake forward/backward retained the declared
shapes and relation. Scalar byte views retained the selected addresses. No
numerical commands ran. The provider guarantees described above remain open.

Python compressed cuts and explicit selections now use the same block-gather
relation. Repeated columns compose through each butterfly source group. Folded
basis weights remain in their existing storage, with selected columns carried
as views. The FX declaration and native matrix preparation share their product
arguments and numerical kernel. Affine strides come from the placed operand,
so a placement change cannot leave a captured stride in the numerical body.
Mapped operands use that same matrix loader's cooperative fragment.

Preparation compiled selected folded transforms at small and large row counts,
and selected butterflies whose basis exceeds one local group's storage. FX
retained the source groups and view relations; flattening preserved them too.
Mapped and newly placed contiguous operands compiled from the same matrix
declaration. No numerical work was submitted.

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
`TensorView`, plus their format and ordered logical column selection.
Stored FFNs, stacked products, experts and conversion kernels use that operand.
Resident row and column cuts compose on it without changing the stored codes.
File imports return this same operand. They retain boundary block bytes, selected
lanes and expert rows through numerical lowering. Padded storage rows do not
become logical tensor rows. The separate buffer-only importer and projection
format/offset fields have been removed. Conversion from component-byte views
to typed element views uses the same layout algebra, preserving mappings.
The shared index emitter retains grouped layouts as arithmetic when a cut starts
inside a group; it does not introduce a lookup table for that phase.

The numerical byte view composes each selected column with its encoded block
during preparation. The decoder reads that address directly. It does not use a
column-table result to index another block table. Its lane relation supplies the
position within the encoding. The shared emitter states where consecutive lanes
form runs, so contiguous and shifted selections retain vector decoding. Irregular
selections use the same decoder with a single lane; no decoded-weight copy is
needed. Scalar formats compose the entire relation into their byte view.

Stacked products retain each binding's own relation. They no longer inherit one
matrix's pitch or first lane, or exclude fusion because those differ. Conversion
uses the same source operand and a declared destination view. Preparation
instantiates each requested format operation with those bindings; the former
bulk instantiations with absent operands are removed.

Preparation on 2026-10-29 built rank and Goal A and compiled stored products,
tiled products and conversions for every supported scalar and quantized format.
Their declarations covered contiguous, shifted, reordered and repeated columns,
partial tails and independently selected stack bindings. Shared layout indices
retained the selected encoded bytes. No numerical commands were submitted.

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
