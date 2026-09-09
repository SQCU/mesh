# Mesh application ABI

The application boundary is the POSIX shared-memory region declared by
[`rdma/mesh.h`](../rdma/mesh.h). The bridge creates and registers that mapping; an
application maps the same bytes. Applications do not open a verbs device, construct a
queue pair, or perform the out-of-band connection.

This document specifies the implemented ABI. It contains no proposed callbacks or
workload-specific transport objects.

## Region and page interface

`mesh_attach(ctx, name)` maps a named region and records its inode. `mesh_open` attaches
the process-global context and returns the first application-arena slot while writing:

- `nslots`: application-arena slot count;
- `stride`: byte distance between slots;
- `usable`: payload bytes in each slot after the wire header.

The low-level submission and completion operations are:

```c
void  *mesh_open(size_t *nslots, size_t *stride, size_t *usable);
size_t mesh_write(const void *p, size_t nbytes, int node);
size_t mesh_write_copy(const void *p, size_t stride, size_t bytes,
                       size_t nslots, int node);
size_t mesh_queue_copy(const void *p, size_t stride, size_t bytes,
                       size_t nslots, int node);
size_t mesh_pump(void);
size_t mesh_queued(void);
size_t mesh_inflight(void);
size_t mesh_read(void **p, int *from);
size_t mesh_readv(void *p, size_t stride, uint32_t *sizes,
                  int *from, size_t count);
```

`mesh_write` submits consecutive arena pages already owned by the application and
returns the payload byte mass accepted in that call. `mesh_write_copy` copies a strided
row array into currently available arena credits and returns the accepted row mass.
`mesh_queue_copy` additionally retains all rows not immediately accepted in a
dynamically growing process buffer. `mesh_pump` advances that buffer and reports queued
plus in-flight row mass.

`mesh_read` returns a pointer directly into one received shared-memory page. Its next
call releases the previous page, so the pointer lifetime ends at that call. `mesh_readv`
copies up to `count` completed pages into a caller-owned strided array and returns their
individual byte lengths and source-node identities.

The `SUB`, `CMP`, `REL`, and `ACK` rings are credit and scheduling structures. Their
depth is not a maximum tensor extent: producers repeat non-queued writes, use the
growing queued path, or use the stream state machine until the entire datum moves.

## Arbitrary-extent stream interface

The implemented stream is a caller-owned `struct mstream` advanced by `mesh_turn`:

```c
void mesh_yell_start(struct mesh_ctx *ctx, struct mstream *stream,
                     const void *source, size_t bytes, int node, uint32_t id);
int mesh_lissen_start(struct mesh_ctx *ctx, struct mstream *stream,
                      void *destination, size_t bytes, uint32_t id);
int mesh_turn(struct mesh_ctx *ctx, struct mstream **streams, int count);
int mesh_turn_window(struct mesh_ctx *ctx, struct mstream **streams, int count,
                     size_t data_window);
int mesh_scatter(struct mesh_ctx *ctx, struct mstream *streams,
                 const void *source, size_t bytes, const int *nodes,
                 int count, uint32_t first_id);
int mesh_gather(struct mesh_ctx *ctx, struct mstream *streams,
                void *destination, size_t bytes, int count,
                uint32_t first_id);
```

`mesh_yell_start` closure-converts a source pointer, total byte extent, destination node,
and stream identity into send state. `mesh_lissen_start` binds the same identity and byte
extent to caller-owned destination storage. `mesh_turn` advances any number of these
states together. DATA frames carry literal byte offsets. FIN reports the full extent,
REQ reports the first missing page offset, and OK closes the sender. Arrival order does
not define placement or completion. FIN is emitted promptly after the data extent.
Unanswered FIN retries back off from 1 ms to 1 s; a repair request or new DATA progress
resets the backoff. Local NIC completion is not evidence that the remote application has
processed a FIN. Retrying at the NIC completion rate produced a control-traffic storm in
the September 5 tensor-parallel measurement. The timer delays retries, not first sends,
ACK processing, receipt, or newly requested repairs.

DATA only changes running receive states. Completed receivers retain enough terminal
state to answer duplicate FIN without reading the freed bitmap. Late REQ cannot revive
a completed sender; failed streams do not consume protocol frames. Caller-owned state
must remain available while its peer may still be completing the exchange. Receiving
MS_DONE alone is not a distributed barrier or a durable tombstone for a retired identity.

Each progress call consumes at most 256 received frames before advancing send states and
returning. This keeps an arriving stream from indefinitely withholding application/GPU
progress. `mesh_turn_window` also admits new DATA only while the context's outstanding
page count is below `data_window`. Control messages can still advance. The original
`mesh_turn` calls it with `SIZE_MAX`, preserving the original DATA-admission behavior.
Neither the receive budget nor the DATA window bounds the total stream extent; callers
continue progressing the same stream until completion. A window of zero pauses new DATA
while allowing receive/control/completion progress.

The `mstream` size and relevant field offsets are unchanged. Sender retry interval shares
the receiver-only `hole` storage, and the former internal `fin_ack` member is now named
`fin_after_ns`. Native callers that inspected that member should rebuild. No wire frame
or shared region layout changed.

`mesh_scatter` and `mesh_gather` divide one contiguous byte extent into adjacent shards
whose union is the original extent. They create stream states only; the caller advances
the complete set through `mesh_turn`. The number of shards is a placement choice, never
a statement about the maximum row or tensor count.

The blocking `mesh_yell` and `mesh_lissen` wrappers use the same state machine and accept
`size_t` extents.

## Bridge progress and receive credits

The bridge sleeps on an idle pass only when no application owns the region. A live owner
can enqueue dependent work at any time, so an empty SUB/CQ pass is not evidence that the
application is idle. Active ownership spends a bridge CPU thread; applications should
release ownership when their session ends.

Receive posting reserves completion-ring credits:

```
q = CMP.head - CMP.tail
q + nRECV <= MESH_RING
receive: (q, nRECV) -> (q + 1, nRECV - 1)
consume: (q, nRECV) -> (q - 1, nRECV)
post: q + nRECV < MESH_RING
```

This preserves capacity and prevents a posted receive from requiring an unavailable CMP
slot. It does not turn UC into a reliable transport: receive exhaustion can still lose
packets, and stream FIN/REQ recovery remains necessary. Following an authorized Mini
reboot and the registration repair below, the final stream/credit implementation passed
735 byte roundtrips and 7,680 distributed projection rounds on M5 Max / M4 Pro with zero
bridge errors. These short cohorts do not estimate service uptime. The existing lifecycle
harness checks stream replay and bounded progress under ASan/UBSan; it does not simulate
the closed-source driver.

## Registered address boundaries

The shipping Thunderbolt provider encodes the low 32 bits of an SGE address beside its
memory-region key. Registration extents must remain within one 4 GiB virtual-address
bank. Splitting into 1 GiB lengths relative to an unaligned mapping does not ensure this:
a receive was observed landing exactly 4 GiB below its requested address. The evidence
and recovery sequence are in [the kernel recovery record](RDMA-KERNEL-RECOVERY.md).

The bridge now splits at aligned 1 GiB virtual-address boundaries. It retains the full
contiguous shared-memory span, uses ordinary `ibv_reg_mr` and absolute SGE addresses,
and shares the same address-to-key lookup between send and receive. Only the first and
last registration may be partial; no page is dropped or capacity reduced.

```
C = 2^30
b = address(mem), h = b mod C, s = span
n = ceil((h + s) / C)
lo(i) = max(0, i*C - h)
hi(i) = min(s, (i+1)*C - h)
MR(i) = [b + lo(i), b + hi(i))
key(a) = MR(floor(a/C) - floor(b/C)).lkey
```

The existing full-span fixture checks unaligned coverage past 4 GiB, key selection,
and reconstruction from region bank plus descriptor address bits. This catches a
mapping-placement defect that ordinary small allocations or output metrics may miss.

## Copy accounting

RDMA receives land in the registered shared region. `mesh_read` exposes that landing
page without a second copy. All other copy boundaries are explicit:

| operation | application-side copy |
|---|---:|
| arena-backed `mesh_write` | none |
| `mesh_write_copy` | source row into arena page |
| `mesh_queue_copy` | source row into pending storage, then arena page |
| stream send | source page extent into arena page |
| stream receive | received page into destination offset |
| Python `Mesh.read` | received page into the reusable NumPy batch |

These alternatives are semantic peers with different ownership contracts. A caller may
choose an arena view to minimize copies or a copied/queued path to retain independent
storage. No documentation may describe a copied path as zero-copy.

`Mesh.read` yields borrowed rows from its reusable 1,024-row NumPy batch. Parse each
row before advancing the generator, or copy it when retaining it. Collecting the
whole generator with `list` or `extend` preserves aliases that the next native
batch overwrites. This stalled full-state RDMA snapshots while the socket fixture
passed; the responder now feeds reassembly directly from the iterator. See
[the real RDMA validation](../measurements/rdma-training-20260906/README.md).

## Restart continuity

The library periodically compares the mapped region inode with the current named region.
A bridge replacement at the same size is remapped at the same virtual address. Page
operations can then continue against the new region. In-progress stream states are marked
failed because the replacement no longer possesses their peer protocol state; the caller
reissues those byte extents. A link re-pair inside one bridge lifetime retains the region
and does not create that boundary.

The ABI data flow is specified alongside the bridge control flow in
[`bridge-and-ipc.md`](bridge-and-ipc.md). Workload tensor framing belongs above this ABI
and is specified in [`ALGORITHM-CONTRACTS.md`](ALGORITHM-CONTRACTS.md).

## Xonotic strategy frame kinds

[`rdma/xonwire.def`](../rdma/xonwire.def) is the single numeric definition consumed by
the Python frame producer/consumer and the DarkPlaces relay. It names observations,
carts, events, strategy responses, expert inference, expert training forward passes,
input and parameter gradients, and batch begin/commit responses. The engine relay sends
every expert request/control kind to the worker and sends every worker response kind back
to its literal source node; it does not interpret or rewrite tensor values.
