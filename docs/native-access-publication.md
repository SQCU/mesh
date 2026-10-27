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
| `StreamedEpilogue.operand` | The standalone command's exact ranges, preserved as indirect accesses |

Packed weight declarations use the same format block size and configured row group as the decoder.
A pitched column share does not read the spaces between its storage groups. A fused epilogue's address
operand keeps each range but drops the old kernel's argument index. Copying that index would attach the
range to an unrelated argument of the fused product. Nonbinding residency declarations remain present.

Some accesses still depend on input values. Expert routing reads selected indices; KV writes use the
supplied page table. Their current declarations still cover the eligible storage where the builder
does not provide a smaller prepared set. Treating a guessed selection as that set would omit real
dependencies. Remaining native builders also need their views supplied. This is not a claim that every
native command now has an exact access declaration.

## Tile publication

The WRITE compiler accepts `ncclMeshWriteSourceReady(plan, word, value)` during preparation.
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
| Native `mmb_heads` | Finishes a head's values, then joins the rows of that head and publishes its prepared word. Fused and standalone heads use the same routine. |
| Native `MeshProduction` | Names a tile width, readiness view and producer command position. The recorder issues the prepared writes before the producer. |
| Native `MeshConsumption` | Binds each WRITE's incoming channel records to its consumer's tile, in that invocation's operand slot. There is no whole-crossing completion command. |

Native query gathers and key/value row handoffs use head tiles. Each source and recipient retains
its own row pitch; relays retain their own pages. All head plans of a crossing are issued as one
group. The declaration and physical plan enumerate the same pieces. Per-invocation readiness
words are ordinary recorded operand slots, including when an epilogue is fused into a product;
no captured address is expected to follow a changing slot.

Source publication is part of the producer. It adds no host submission, peer acknowledgment,
channel-prefix scan, or spin waiting for another producer threadgroup. Arrival counts use the
existing `mesh_last_arrival` helper. Producer stores exposed during a dispatch are system-coherent.
The native epilogue's contributor count clips its final span to the actual output width.

## Tile consumers

`ncclMeshWriteLandings` returns each WRITE's final incoming prefix on each used physical
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
final positions. The producer publishes each head after its row contributors finish, and
each merge reads only that head's contributors. The old native `partials` all-reduce branch
is removed; old records must be derived again. This changes collective decomposition as
well as scheduling; end-to-end latency and useful throughput still need measurement.

The DNN fold retains its row-tile dependencies. Native source publication is still one
head across its rows. Local attention kernels still finish their own dispatch before the
separate partial-result producer executes. The PLE epilogue has no declared remote output.

A channel still has FIFO ordering. Independent readiness words remove the false dependency
on every producer tile finishing; they do not make a later descriptor overtake an earlier one.
The routing compiler can choose a different order using the same interface.

## Validation

The shared library and Torch extension built. The matrix producer prepared a 131 by 257 by
259 product with row tiles of 64, 64 and 3, without numerical submission. The native/LiteRT
rank built; Metal compiled the standalone and fused native head producers. The existing FX
transformer forward/backward graph also prepared at batch 2, sequence 1,024 and width 2,048.
The consumer refactor additionally compiled the native vector, tile and shared-prefix
attention instances, partial-result producer/merge and cache writer for macOS 26. No
distributed numerical run was launched. These results do not establish an end-to-end
latency/useful-throughput improvement.
