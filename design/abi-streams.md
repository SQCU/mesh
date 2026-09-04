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
not define placement or completion. FIN is emitted immediately after the data extent and
is retried only after the preceding FIN page has a transport completion; no spin count,
elapsed-time window, tensor shape, or workload count controls stream progress.

`mesh_scatter` and `mesh_gather` divide one contiguous byte extent into adjacent shards
whose union is the original extent. They create stream states only; the caller advances
the complete set through `mesh_turn`. The number of shards is a placement choice, never
a statement about the maximum row or tensor count.

The blocking `mesh_yell` and `mesh_lissen` wrappers use the same state machine and accept
`size_t` extents.

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
