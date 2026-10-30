# Fake guards and self-imposed waits in the runtime, 2026-10-30

The rule: a check is merited at most once, where its fact is made. A compiled program's facts (spans, lengths,
frames, owners, lifetimes, orders) are checked by the compiler that makes them, at compile time, by the producer
(design/rubrics.md, compile-time checks; the operator: "checks at runtime instead of at compile time are forbidden",
"trace a malformed message to its producer"). A check of the same fact again at run time is a fake guard. A wait the
compiler did not deduce from a data dependency or a hardware limit, and choose, is self-imposed. The runtime is
waitless but for the waits compilation deduced and chose, and those are reduced to the fewest waits and the least
pipeline stall the dependencies allow.

The case that shows it: a relay, a node forwarding a page it received unchanged so its neighbours' inbound cables
fill. Its data is not touched; every cost on its path is overhead.

## The bridge: what a relay unit pays (rdma/mesh-link.c)

Per unit sent (stream_send) and per receive posted (stream_post), on literally unchanged bytes:

| Cost | Where | Kind |
|---|---|---|
| `mesh_dma_span_valid` of the unit's span | stream_send, stream_post | fake guard: the library validated every span when it made the table (place_group), and walk_take validated the table |
| `mesh_entry_table_valid` of each entry taken | walk_take | fake guard: the table is the library's compiled output |
| `!e->whole` (the entry's word in the region) | stream_send, stream_post | fake guard: fixed per entry, compiled by the library |
| a unit's frames against the queue's capacity (EMSGSIZE) | stream_send, stream_post | fake guard: a span's length is compiled at most a page |
| `unit_told`: four seq_cst stores of the unit's interval (mesh_pin_store), a seq_cst store of `posted`, a seq_cst load of the owner (holding_kept), an undo where it moved | stream_send, stream_post | fake guard: re-asks per unit whether the client still holds the stream; the library retires a lease only after its last unit completed (mesh-lease-retire, compiled lifetimes) |
| `metadata_pin`: five seq_cst stores and a load each time the walk reaches another entry | stream_send, stream_post | fake guard, the same question per entry |
| one `ibv_post_send` and `IBV_SEND_SIGNALED` per unit, one completion processed per unit | stream_send, link_send | self-imposed: a doorbell and a completion a unit where a chain of requests and selective signalling need one a batch |

Forced, deduced at compile time and kept: the unit's readiness (the relay's source `landed`, a send's `ready`), the
send gate's frame credit (TN3205: a queue accepts posts past its frames and corrupts later), a receive posted before
its SEND arrives (the hardware's credit).

## The GPU and the host (rdma/nccl-mesh*.{c,m,h}, rdma/metal-recording.m)

| Cost | Introduced | Kind | Now |
|---|---|---|---|
| a pool of up to 4096 spinning workers | 4530374d | self-imposed | at most one a core, d946e1e4 |
| a system-scope fence for every cursor a worker checks | 4530374d | self-imposed | one a pass, 4ed1a7f3 |
| every receive and SEND waiting for the whole consumption of every earlier overlapping receive | d3869552 | self-imposed: the dependency is per element | chunk by chunk between matching combines, f1a821ab |
| a combined SEND published once its whole landing is summed | bd83c006's revert of 63a3fe3d | self-imposed | unit by unit, 657f68f8, cd210504 |
| a full queue barrier before every call (barrierAfterQueueStages), so call k + 1 presents only after call k's pool ends | 20afa368, 8fa0b005 | self-imposed: the dependencies are per channel and per buffer range | open |
| the routes solved inside planning, the program waiting for them | 8e5a631c | self-imposed: a solution is an operand, k-buffered in pages, solved apart from the program; no new solution means the same values in the same pages | open |

## The front end

| Cost | Kind |
|---|---|
| the completion observer looking every 5 ms (rdma/rollout.py) | the measurement's own resolution, not the program's; superseded for cables by `mesh-stat landed` |
