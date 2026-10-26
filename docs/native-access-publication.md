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

## Where tiles become available

The existing numerical completion mechanisms differ from their transport bindings:

| Producer | Existing completion point | Current transport boundary |
|---|---|---|
| `matrix.metal` `product` | One arrival counter joins every row, column and batch threadgroup | All source publications released by the final threadgroup |
| DNN `prepare` | Configured outer row tiles, each with its own product and fold | One whole crossing per outer tile; fold observes its full incoming prefixes |
| Native `mmb_after` | `mesh_last_arrival` joins only the contributors to an epilogue span | Finishing a head or block does not yet release a prepared WRITE |
| Native `CrossingPoint` | A recorded crossing mark after its producing commands | The ordinary library publication command at that mark |

`mmb_after` already has the useful numerical structure: a span's last contributor runs its epilogue.
Other threadgroups do not spin there. `mmb_heads.span()` names a head width; `mmb_ple.span()` names
its block width. A split product must finish all contributors to that span before nonlinear operations
or a final-value send. An access range alone does not describe those incomplete sums.

## The next compiler boundary

The compiler needs to connect each produced span to its source page/mask list, prepared physical writes
and consumer views. The producer then releases that span's existing transport bindings after its stores
are visible. The consumer names the incoming data it needs. Relays retain their own incoming-data
dependencies. This uses the prepared write compiler and all its physical channels.

The current `ncclMeshFused` interface returns one cumulative publication per active channel for the
whole crossing. Its repeat index describes an ordered train of frames. It is not an independent tile
identifier: releasing frame 3 also certifies the earlier entries of that channel. GPU threadgroups can
finish out of order. Releasing their numbered prefixes independently would advertise unfinished data;
an unconditional later store of a smaller prefix could also move readiness backwards.

The unresolved integration is therefore the compiled relationship between production order and these
physical prefixes. It must expose tile completion without a new host submission per tile, a peer receipt,
or a GPU spin waiting for an unscheduled producer. The current whole-dispatch join has not been removed
by the access-declaration change. The existing span completion, page plans and common schedule are the
components to connect; another execution interface is not needed.

## Validation

This change alters preparation metadata. Numerical bodies, launch grids, write plans and publication
points are unchanged. The shared library and Torch extension built. The existing transformer block
prepared its complete forward/backward graph at batch 2, sequence 1,024, width 2,048 and float16 using
the shared range calculation. No numerical submission was made. This does not establish an end-to-end
latency or useful-throughput improvement.
