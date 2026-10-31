# Native accesses and tile publication

Reviewed 2026-10-26. This follows the compiler integration in
[compiler-pipeline.md](compiler-pipeline.md). The data-model document is unchanged.

## Access declarations

A command declares which bytes it reads and writes. The compiler uses those ranges to order commands
that share data. Naming every byte after an argument's start can make separate slices appear to share
data and add barriers between them. Narrowing that declaration does not by itself publish an early tile.

`MetalViewReaches` in `rdma/metal-recording.m` now lowers shape, byte strides and element width for both
FX's `_stream.Commands` and Swift's `Reach.view`. Dense dimensions coalesce; transposes, broadcast
dimensions, reversed strides, gaps and empty views retain their actual access union. This calculation
runs at preparation. Transport's logical byte order is the separate C compiler `mesh_view_layout`
(`rdma/mesh-write-compiler.h`; `mesh.view_layout` from Python).

| Native command | Declaration now supplied |
|---|---|
| `rmsNormCommand` | Input and output matrix views, including column strides and grouped rows; gamma vector |
| `rmsNormAddCommand` | Input, residual and output rows; gamma and optional scalar |
| `scaleCommand`, `addCommand`, `geluCommand` | Each operand's own offset, row width and pitch; correct read/write role |
| `formatProduct(.rows)` | Input rows, packed weight blocks and directly placed output rows |
| `stackCommand` | Each slab's own weights and output; unused argument slots declare no bytes |
| `headsCommand` | Query columns at their final offset and pitch; KV output rows |
| `LayerCaches.reaches` | The prepared local cache page sets for the layer, intersected with each backing chunk |
| `attentionCommands` | Query, fresh KV and partial-result views, row words, tile landings and publication words |
| `streamedInput`, `streamedFinish` | Input/output row views and the partial-sum planes they actually use |
| `StreamedEpilogue.operand` | The standalone command's exact ranges, preserved as indirect accesses |

Packed weight declarations use the same format block size and configured row group as the decoder.
A pitched column share does not read the spaces between its storage groups. A fused epilogue's address
operand keeps each range but drops the old kernel's argument index. Copying that index would attach the
range to an unrelated argument of the fused product. Nonbinding residency declarations remain present.

Some accesses still depend on input values. Expert routing reads selected indices; KV writes use the
supplied page table. Their current declarations still cover the eligible storage where the builder
does not provide a smaller prepared set. Treating a guessed selection as that set would omit real
dependencies. The cache declaration covers its prepared eligible pages, not a guessed selection of pages from future input values.

## Tile publication

The WRITE compiler accepts `ncclMeshWriteSourceReady(plan, words, value, layout)` during preparation.
The source's SEND descriptors use that word instead of the channel's cumulative ready prefix.
A tile's producer calls the shared `mesh_source_release` after its stores. Each tile has its own
word, so a later tile finishing first cannot release an unfinished earlier tile. Wire order is
still the compiled order. A relay's descriptors still depend on its incoming landing word.

The target covers the tile's logical bytes. Each physical fragment derives its unit interval
below that target at preparation. Publication releases the fragment's existing page/mask list;
it does not pack, copy or construct descriptors. The readiness operand is retained through the
bridge's final CPU read by the same lease ledger as the descriptor list. That ledger runs in
preparation and retirement, outside the numerical function.

| Producer | Completion and binding |
|---|---|
| `matrix.product` | A separate arrival count for each batch/row tile joins only that tile's column groups. Its last contributor releases that tile's source word. |
| DNN `prepare` | Compiles the same `matrix.row_tiles` into WRITE plans and row-tile consumer bindings. The product and fold remain whole dispatches. |
| Native `mmb_heads` | Finishes one row/head and publishes that row/head’s prepared word, without joining other rows. Fused and standalone heads use the same routine. |
| Native `MeshProduction` | Names a tile width, readiness row stride, view and producer command position. The recorder issues the prepared writes before the producer. |
| Native `MeshConsumption` | Binds each WRITE's incoming channel records to its consumer's tile, in that invocation's operand slot. There is no whole-crossing completion command. |

Native query gathers and key/value row handoffs use head tiles. Each source and recipient retains
its own row pitch; relays retain their own pages. One WRITE plan describes each sender/recipient pair. The crossing issues those plans as one
group. The declaration and physical plan carry the same tile boundaries. Per-invocation readiness
words are ordinary recorded operand slots, including when an epilogue is fused into a product;
no captured address is expected to follow a changing slot.

Source publication is part of the producer. It adds no host submission, peer acknowledgment,
channel-prefix scan, or spin waiting for another producer threadgroup. Arrival counts use the
existing `mesh_last_arrival` helper. Producer stores exposed during a dispatch are system-coherent.
The native epilogue's contributor count clips its final span to the actual output width.

## Tile consumers

`ncclMeshWriteLandings` fills each tile's final incoming prefix on each used physical
channel. Multiple tiles remain in one group; recording a tile adds no host submission.
Relay-only entries are not consumer inputs. Native `TileLanding` stores these lists in the
same invocation slots as other recorded operands. The shared `mesh_landing_counts` and
`mesh_landing_await` helpers observe the lists without another network message.

Vector attention observes its query head. Tile and shared-prefix attention observe only the
query heads in their active rows. Fresh KV rows are read directly from their received views;
each page that intersects those rows observes its own KV head's incoming lists. Old KV pages
continue through the cache page table. The shared attention template accepts this read layout
as a compilation operand; its ordinary cached layout still uses the existing matrix loads.
Cache installation follows attention and uses the same per-head dependencies. Handoff-only
receivers use that same cache writer. No consumer waits for another GPU threadgroup to run.

The native softmax configuration now uses the existing gather-and-merge algorithm. Its
partials occupy `[row, contributor, head, m/l/O]`, so head columns land directly at their
final positions. Attention's last contributing part writes that result and publishes each
row/head as it finishes. The separate partial-packing dispatch is removed. Each receiving
merge reads only that head's contributors. The old native `partials` all-reduce branch
is removed; old records must be derived again. This changes collective decomposition as
well as scheduling; end-to-end latency and useful throughput still need measurement.

The DNN fold retains its row-tile dependencies. Native source publication is one row/head.
Masked rows publish their own empty partials within the same producer.
The PLE epilogue has no declared remote output.

A channel still has FIFO ordering. Independent readiness words name each tile's data.
The compiler orders descriptors from the producer records described below; a running
descriptor list does not choose another tile when its next tile is unfinished.

## Earlier compile evidence

The shared library and Torch extension built. The matrix producer prepared a 131 by 257 by
259 product with row tiles of 64, 64 and 3, without numerical submission. The native/LiteRT
rank built; Metal compiled the standalone and fused native head producers. The existing FX
transformer forward/backward graph also prepared at batch 2, sequence 1,024 and width 2,048.
The consumer refactor additionally compiled the native vector, tile and shared-prefix
attention instances with fused partial publication, the receiving merge and cache writer
for macOS 26. The complete Swift/native/LiteRT rank build succeeded. No
distributed numerical run was launched. These results do not establish an end-to-end
latency/useful-throughput improvement.

## Integration on 2026-10-26

`MeshProduction.rowStride` and `MeshConsumption.rowStride` give each row/head its own
prepared readiness word and landing record. The common group contains all of them. Query
consumers name their own row/head; a fresh KV page names only the rows it contains.
Head publication no longer allocates or increments a cross-row arrival counter.
Attention’s reduction still combines the numerical partials required for that row.

Page masks are now the public WRITE declaration in `nccl.h`. Tensor-view adapters
live in `mesh-write-compiler.h` and resolve local views into that page domain during
preparation. They preserve the element cuts of an already declared typed call.
Adjacent page selections coalesce before physical descriptor generation.


## Compact publication plans — 2026-10-27

For a crossing with `I` invocations, `R` rows, `H` heads per source and `P` active
sender/recipient pairs, the former native builder made up to `I × R × H × P`
WRITE plans and repeated the same number of call declarations. Each tile plan
exchanged placement metadata and compiled routes separately.

`columns_prepare` now makes `P` plans. The tensor compiler's `mesh_write_layout`
supplies the logical tile width and each rank's payload-slot count and byte stride.
The public page API's `ncclMeshWriteLayout` uses strides in local pages instead;
its masks stay within one page. Both lower through the same preparation function. The source and
recipient keep their recorded binding periods. A relay gets distinct payload
slots for the function's invocations; sharing a plan does not reuse its data
early. The packet compiler derives common cuts from the peers' slots for that invocation
only. Packet sizing restarts at each publication tile. An occurrence never inherits
another occurrence's registration cuts. An invocation indexes its prepared descriptor train when commands are
recorded, before numerical execution.

`mesh_write_compile` allocates the complete pair's flow across its configured
paths, then cuts every path at publication boundaries. Each resulting fragment
names its tile's source word; relay fragments still name incoming landed words.
`ncclMeshTileLayout` maps logical tiles into readiness and landing tables. Row
pitches, table strides and invocation periods are explicit local operands.
A tile split across several cables retains all of its required landing records.

`ncclMeshCommPlan` compiles each group definition once. `ncclMeshPlanRun` describes
contiguous group blocks and their repeat counts. Native classes supply definitions;
the function's run order references them. Useful-byte demand includes every block
occurrence. The former expanded array of issue pointers is also removed. Python
and Torch declaration callers use the same interface.

Dependency words, landing records, and actual DMA descriptors still scale with
the independently live tiles and payload bindings. Those are data relationships,
not duplicate routing problems. Numerical submission remains `MetalScheduleEnqueue`;
there is no new tile loop, route lookup, allocation, or host wait in that submission.
Item 5's ordering change follows below. The shared stream still limits a group to its
configured `MESH_STREAM_ENTRIES`; plan compaction does not remove that physical-list limit.
Compact affine/page lowering is recorded under item 8 below.

The shared libraries, CFFI module, Torch extension and macOS 26 Swift/native/LiteRT
rank compile. No numerical execution or performance measurement was made for this
change. These builds establish integration of the changed interfaces, not a
latency or useful-throughput result.

## Producer order in the common WRITE compiler — 2026-10-27 06:47 UTC

The former `ncclMeshCommWriteRelease` supplied one predicted time per rank.
All tiles from that rank therefore looked equally ready. Ties followed the
transfer array's payload order. A native head grid traverses rows within a
head, while its output stores heads within a row. For two rows and three heads,
the grid's order over logical output tiles is `0, 3, 1, 4, 2, 5`, rather than
the memory order `0, 1, 2, 3, 4, 5`.

That rank-wide interface is removed. `ncclMeshCall.production` supplies one
`ncclMeshProduction` record per logical tile: predicted seconds relative to
the group's producers and the producer's ordering ordinal. Native source ranks
report their own records during graph realization. The merged declaration gives
every rank the same source records. Python, MPI and Torch declarations consume
that same C interface. Missing tile records fail preparation.

Native head lowering supplies its row/head traversal, including shared K/V
production. Attention supplies its vector traversal or its KV-head-group
traversal. The current native seed declares zero time differences and uses these
ordinals; it does not invent measured tile times. Masked rows and GPU scheduling
can finish in a different order. Time predictions are compilation operands, not
runtime delays or permission to send unfinished data.

`mesh_write_order` combines those records with the selected physical paths,
link capacities and streamed parent dependencies. It chooses the earliest
predicted available cable slot, then the producer ordinal. Source and recipient
use the same global ordering.
Relay fragments retain their incoming-page dependencies. All WRITE groups now
use that one schedule. The second `ordered_writes` copy, `write_plans` storage
and fused/unfused schedule selection are removed.

Fragment lookup uses a sorted index. Each cable's compiler heaps hold pending
and predicted-ready fragments; each fragment enters and leaves each heap once.
The order builder scans cable candidates rather than every remaining fragment.
For `T` fragments and `E` directed cables, this ordering stage takes
`O(T log T + T E)` work and `O(T + E)` storage, replacing its quadratic scans.
These temporary compiler structures are absent from execution.
Relay dependency construction also joins the compiler's disjoint, ordered
incoming and outgoing tile intervals in one pass. It no longer compares each
WRITE tile with every earlier tile.

The DNN also declares one tiled WRITE family per peer pair and layer, rather
than one group per row tile. Its matrix launch supplies row-tile order. The
existing `fused.writes` adapter binds the family through the same page compiler;
the fold still consumes individual tiles. Contributors can share a consumer's
landing table. During preparation, the compiler reduces their records to the
last required prefix on each physical cable. This removes duplicate dependency
checks for that tile. The final short tile publishes the family's marker while
its page masks still transfer only the useful bytes.

These changes expose production order to route compilation and remove the
per-tile group boundary. They do not solve whole-function empirical fitting or
make a FIFO overtake unexpectedly late work. Section 7 still has to connect
optimizer operands and complete-call observations. The physical group-entry
limit remains explicit. No runtime scanner, host callback, ACK, payload staging
or additional submission boundary is introduced.

The C libraries, CFFI module, Torch extension and macOS 26 native/LiteRT rank
build. Python source parses. Numerical execution and performance measurements
remain deferred until the structural sequence is complete.

## Explicit native binding views — 2026-10-27

Source changes recorded 2026-10-27 16:31 UTC, structural item 6.

`MetalAccesses` no longer chooses the remaining allocation or an invocation
stride when a binding has no extent. Dense extents include zero; strided binding
reaches supply their union of ranges. Strides only relocate those ranges across
prepared invocation slots.

Native `MetalDispatch` builders supply dense `reads`/`writes` lengths or existing
`Reach` views. Both become the same records before execution. Construction checks
that each used, non-null buffer argument has a declaration. `MetalProgram.bind`
uses the declared access modes rather than assuming every operand is read and
every unreflected operand is also written. The separate raw Swift `encode`
helper is removed. Its serving copy and weight-conversion callers use the common
prepared command interface.

Declarations now include routed expert storage, control operands, sampling,
descriptor copies, drafting, vision and DSV41 commands. Shape operands determine
payload extents. Runtime-indexed tables declare their eligible domains; an input
value cannot be guessed during preparation. Immutable address and control tables
may be read in full as named operands. That does not make the containing payload
arena an access.

Coded products use the immutable tile and band words to find the selected code
planes and input slabs. Their field masks and plane width come from
`etc/compute/metal.def`, also consumed by the Metal decoder. Column pitches exclude other column slices. Finished products use
the supplied row order to name their output columns, including permutations with
gaps. Head consumers declare their selected Q/K/V source columns and partial-sum
planes. Fused epilogues retain their standalone declarations; the former
buffer-to-end substitution is deleted. Publication words keep their binding
indices when passed to the fused producer.

Recapturing a prepared command preserves its dense constants, empty views,
strided ranges and explicit read/write modes. Previously constants lost their
lengths at this boundary, and reflection could restore a broader access mode.
All view construction and missing-input errors occur during preparation. No view
lookup, allocation, new wait or validation is added to numerical submission.

The raw recorder requires views too. LiteRT native stages supply the logical
`MeshOperand.bytes` for each bound invocation. Partial-row export and merge
stages carry explicit provider layouts; the shared `MetalTensorLayout` emits
their shader indexing and access ranges, including the holder's output offset
and the normalizer's skipped vector lanes. External delegate tensors already
supply their typed storage views. The packaged LiteRT delegate does **not** yet
export its internal views: `InferenceContext` places tensors at offsets in shared
buffers, and `MetalArguments::Encode` passes only buffer and offset. Those
internal bindings now fail preparation. Since 2026-10-29 a provider stage is not
recorded at all (its commands are LiteRT's own encoding, which no description holds);
restoring a buffer-tail extent would recreate the defect. This is an outstanding
backend integration requirement before numerical execution, not a completed
LiteRT capability or a performance result.

The shared C libraries, Torch extension and macOS 26 Swift/native/LiteRT rank
build. The streamed Metal source also compiles for macOS 26 with the queried
device capability operands.
No dispatch was submitted. Numerical runs and measurements remain deferred until
the structural sequence is complete.


## Assignment feedback — 2026-10-27

Structural item 7 connects the shared numerical assignment to complete-call evidence and producer ordering.

- `assignment.assign` requires a proposal ordinal independently of the retained observation count. The old
  `len(rows)` schedule stopped moving after the history window filled. Paired directions now keep advancing;
  declared tile and route quanta determine the smallest observable operand change.
- Fixed and work-dependent coefficients are fitted from complete-call durations. The old fit used only optional
  `costs` arrays, which most callers supplied empty. The fit now receives every compatible completed call and
  keeps a regularized uncertainty estimate. A residual fit updates compute, engine, route and compiler shares.
- Late records retain their executed proposal, operands and implementation. Completion bounds come from the
  maximum of peer intervals. The observation worker still collects them without holding up numerical execution.
- Chain/LSTM/DNN grouping operands and the FX product crossover are declared configuration domains and consumed
  during preparation. Native GPU/ANE shares remain domains even when diagnostic timing arrays are empty.
- Torch and native producer records use `ncclMeshProductionTimes`. It consumes the inferred source duration and
  existing traversal; the runtime still sees the same page descriptors and publication words. No timing event or
  per-tile measurement was introduced. Native routes receive the same integer path weights as Python transport.

The model cannot identify every node's isolated compute or hop latency from one complete-call scalar. Its
producer times remain estimates. This source change establishes the feedback and realization path; it does not
establish a throughput/latency gain. Shared libraries, CFFI, the Torch extension and the native/LiteRT rank compile.
Python and configuration documents parse. No numerical workload, test suite or hourly was run.

At this point items 8 and 9 remained: compact page/mask compilation and general backend/coded-weight layouts. The LiteRT internal
view metadata gap recorded in item 6 remains a provider requirement; this change adds no substitute executor.

## Compact page-bound views — 2026-10-27

This dated implementation record predates the 2026-10-30 interface consolidation.
The current preparation boundary is `ncclMeshCommPlan` with Call buffer slots,
pure typed views and progress operands; `ncclMeshCommEncode` records its next
group. Both standalone WRITE preparers and their execution/serialization
lifecycle have been removed. The older API names below describe that revision.

Structural item 8 removes the row → span → page-mask → span round trip. `mesh_compile_rows` now keeps
row width, row count and local pitch as scalar operands. Typed repeated WRITEs no longer allocate a page-mask
array only to discard it. `mesh_compile_spans` retains the caller's declared selections without splitting each
one into pages. The public `ncclMeshWritePagesPrepare` still accepts only page identities and masks confined
to one page; its adjacent selections coalesce during preparation.

All adapters resolve local storage to a page table and page-local start. The common preparer consumes that
domain, bounds the whole selection and its declared slots there, then emits the final physical descriptors.
The mask of a compact row within any page follows from its start, width and pitch. These expressions need no
stored entry per intersected page. No process pointer is exchanged, and no raw pointer WRITE API was restored.

The common cut builder handles different row shapes, pitches and indexed layouts on different peers for the
same logical bytes. An indexed source can therefore land in affine destination or relay storage. The old report
comparison rejected that combination. Each invocation uses its own slot's registration phase; the removed row
planner used the first slot's phase. Cuts also retain logical publication boundaries and their packet grid.
Contiguous row/page/span edges impose no extra cut. Contiguous payloads and typed repetitions keep the existing
modular residue solver, without a per-packet walk across peers.

`mesh_view_layout` (`mesh.view_layout` from Python) keeps affine NumPy and Torch views compact before crossing the native binding. The old
`SpanWrites` binding is replaced by `ViewWrites`, which supplies row operands or irregular selections to the
same native preparer. Its row operands also replace the NumPy adapter's separate affine-layout calculation.
Explicit indexed selections that describe affine rows collapse to a fixed report. Reversed and aliased source
views retain their mapping; overlapping destination selections fail preparation.

Affine layout metadata now has constant size per peer, independent of row and page counts. Common cuts and
final descriptors still scale with physical fragments and independently placed invocation slots. Irregular
lists still scale with their declared selections. No mask walk, shape check or allocation was added to numerical
execution. The separate `mesh-row-span-plan.h` compiler is removed.

Item 9 remains: general backend/coded-weight layout integration. Numerical execution and hourly measurements
remain deferred until the structural sequence is complete.

The shared libraries, CFFI module, Torch extension and native/LiteRT rank compile. Changed Python sources parse,
and the diff has no whitespace errors. No numerical workload, test suite or hourly was run.
