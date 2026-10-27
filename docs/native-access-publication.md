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
runs at preparation. Transport's logical byte order remains the separate `views.byte_spans` operation.

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

A channel still has FIFO ordering. Independent readiness words remove the false dependency
on every producer tile finishing; they do not make a later descriptor overtake an earlier one.
The routing compiler can choose a different order using the same interface.

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
FIFO ordering remains item 5. The shared stream still limits a group to its configured
`MESH_STREAM_ENTRIES`; this change removes duplicate plans, not that physical-list limit.
Compact affine/page lowering remains item 8.

The shared libraries, CFFI module, Torch extension and macOS 26 Swift/native/LiteRT
rank compile. No numerical execution or performance measurement was made for this
change. These builds establish integration of the changed interfaces, not a
latency or useful-throughput result.
