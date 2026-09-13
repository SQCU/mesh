# Streaming algebra with indexed destinations

The substrate represents values as independently usable extents in canonical
registered shared memory. Numerical coordinates, readiness extents and transport
blocks are separate. DNN modules compose the algebra operations; the transport
and algebra headers contain no model definition. The [backend streaming contract](backend-streaming.md)
works through MPS, ANE and symbolic NumPy-style Metal examples, with mandatory
producer and consumer progress in each.

## Paired transfers

`mesh_algebra_copy` takes a source endpoint, destination endpoint, occurrence
count and transport queue. Each endpoint supplies a tensor, peer ID, first extent
and extent stride. Both endpoints are declared together on every participant.
The endpoint tensor is the locally realized instance of the corresponding
distributed tensor; its peer ID selects which participant owns this occurrence.
The destination peer receives into its actual registered tensor pages.

For occurrence i, the endpoints are
`source.tensor[source.first + i * source.stride]` and
`destination.tensor[destination.first + i * destination.stride]`. Shape and
scalar type must agree; a remote transfer also has equal padded extent lengths.
Every participant numbers transport blocks in block-major, then occurrence order
before projecting its local work.
The source peer contributes a SEND binding and the destination peer contributes
a RECV binding with the same block identity. A same-peer transfer becomes
a local affine copy. No tensor payload is packed into another transport store.

All participants execute the same ordered copy declarations with the same global
parameters. Local numerical graphs and physical addresses need not be identical.
The old mesh_algebra_transfer entry point remains available for manually managed
bindings; a paired program uses mesh_algebra_copy throughout so its occurrence
identities have one owner. It does not mix independent manual identities into
that same transfer schedule. The current low-level send binding holds one send
reader per source extent: these maps give each remote occurrence its own source
and destination extent. Repeated transmission of one source extent to several
destinations is not implemented by this interface.

## Storage and execution

Create tensors and bind numerical functions before mesh_algebra_realize. Configuration
allocates the registered pages, zero-copy Metal aliases, layouts, specializations
and matrix bindings. Numerical invocation scans the configured functions with
mesh_issue, submits their GPU commands, and publishes with mesh_complete.
No allocation of operand storage or choice of backend occurs during invocation.

Affine, weighted addition, multiplication, tanh, exp, row sum, reciprocal square
root and MPS contraction are available. Views can slice, transpose and broadcast.
A bound output covers its complete independently allocated extent, and binding
mandatorily decomposes it into independently issued and published parts. Local
parts occupy one page; transferable parts occupy one configured SEND block.
Input dependencies cover only the pages touched by each part's indexed arithmetic.
MPS receives rectangular subproblems covering that part; a part crossing matrix
rows may use multiple rectangular calls in its command buffer. K-panel partials
have separate outputs and explicit additions fix their association.

The pre-realized numerical encoder runs inside emit_part, whose completion
unconditionally calls mesh_complete. That publishes PRESENT and the canonical
send work for the part. Neither the caller nor a collective implementation can
omit this publication through the algebra API. Later parts need not be ready.
mesh_algebra_return_part registers a reader for one such publication part;
whole-extent returns remain available for callers consuming the complete value.
Page granularity cannot expose independently ready elements within one page.
A row reduction still needs the columns it actually reduces.

Mutable external producers call mesh_tensor_issue before writing and
mesh_tensor_complete afterward. mesh_tensor_publish publishes preinitialized
bytes; mesh_tensor_constant declares immutable values before realization.
Registered returns remain readable until mesh_algebra_consume. NIC reads and
numerical readers retain source storage through their existing completion bits.

The implementation follows Pallas's larger-buffer approach: distinct live value
instances receive distinct pages. The example realizes R copies of the intermediate
arrays and shares immutable weights. Invocation i uses slot i modulo R. The next
invocation in that slot is produced after that slot's outputs are consumed;
other slots' outputs can remain live. No collective-wide release is necessary.
Within a window, incoming values have separate destinations and no wraparound.

## Source proof for the acceptance program

Each slot has independent producer, peer contribution, transform, K-panel partial,
result, gathered output, statistic and normalization extents. The example calls
the same configure function for every slot on both participants.

For each slot, producer copies enumerate groups, peers, transport blocks, and
then the two strided K-panel extents. Result copies follow. After those slot
configurations, direct copies of contraction partial zero use queue two. Each
transport block receives a unique identity before either participant projects
its local SEND or RECV. The corresponding FIFO sequences therefore name the
same value and destination regardless of local row/page numbers.
The live slots use disjoint allocations. Reuse is subject to those pages' actual
reader lifetimes; the ring of work-request metadata is not the value-buffer ring.

For each window the caller withholds input zero of its first invocation, while
producing all other inputs into their separate slots. It requires group one's
contraction to complete in every invocation in the window before supplying the
withheld input. With depth eight, seven later invocations must produce that output
while the first invocation still lacks input zero. No outputs have been consumed
at that observation point. For a 257-row input, it next writes and publishes only
rows 0 through 255 of input zero. The first contraction output block must arrive
at the peer and match the independent formula while row 256 remains unwritten.
The full contraction receive must remain unavailable. Only then is the input tail
written and published. Afterward the host consumes and refills one slot at a
time, while the same static scan runs all eligible numerical functions.

Queue zero's FIFO can still block later group-zero transfers behind the missing
first input. Queue one has a separate configured order and carries the independent
work. More memory removes destination-reuse dependencies; it does not make an
anonymous SEND randomly address a later slot on the same queue.

## Build and acceptance

Build `make -C rdma .build/streaming-algebra`. Run
`rdma/.build/streaming-algebra 0 19 257 8` and
`rdma/.build/streaming-algebra 1 19 257 8` on the two peers. Arguments are rank,
invocation count, rows and depth. A 4096-page arena with 4-page blocks and three
queues accommodates this example. Shape 257 exercises extents spanning
multiple transport blocks and partial publication inside a logical contraction. The existing executable is the acceptance path.

The numerical expectation varies by invocation and is computed independently.
Every contraction, peer replica, row sum and composed normalization is checked.
Reports include allocated pages, delayed windows, later-invocation completions,
numerical error, GPU commands and observed early-window time with online count,
mean and sample variance. These intervals include host observation and initial
setup when applicable; they are not throughput gains.

The [earlier measurement record](data/streaming-algebra-2026-09-13.json) concerns
the earlier single-slot implementation. It is not validation of this buffering
change. The [source review](pallas-collective-source-review.md) records why the
paired maps and explicit invocation storage were needed.

## Boundaries

The current bridge connects one peer using two-sided SEND/RECV. It does not
implement a general multi-peer topology. Shared-memory version 20 requires
rebuilt clients. Connection setup waits for realized configuration, posts the
initial receive windows and completes a bilateral setup boundary before sending.
That fixed setup cost does not become a numerical firing predicate.

MPS and Metal bindings automatically publish parts inside the logical algebra
operation. Each underlying command buffer supplies an actual device completion
boundary; mesh does not pretend to call a host sender from inside an opaque MPS
shader. This backend implements output-part publication, whereas the cited Pallas
callback forwards input shards during its matmul pipeline. No ANE adapter is
implemented here; an adapter must preserve the same mandatory partial-publication
contract.
FP16 is supported by the interface but has not been covered by this example.

## Recorded acceptance

[Results](data/pallas-indexed-destinations-2026-09-13.json) cover both participants:

| Commit | Invocations | Depth | Rows | Delayed windows | Later independent outputs per participant | Allocated pages per participant |
|---|---:|---:|---:|---:|---:|---:|
| f17e615 | 32 | 8 | 128 | 4 | 28 | 964 |
| 2cf2fb9 | 19 | 8 | 257 | 3 | 16 | 2276 |

The second run uses paired strided maps for both K panels, crosses transport-block
boundaries, wraps invocation slots, and ends with a partial window of three slots.
All numerical comparisons pass; maximum contraction absolute error is
1.38101313e-7. Peer replicas agree exactly. The paired source mapping establishes
which destination owns each occurrence; these runs check numerical composition
and observable progress, not a transport-loss theory or a claimed speedup.
C/Objective-C warning checks and Swift import/typechecking pass.

### Mandatory publication acceptance

[Recorded results](data/mandatory-partial-publication-2026-09-13.json) for
f280de7 use three queues, 19 invocations, eight storage slots and 257 rows.
Both M5 and M4 received and numerically checked the first 256 contraction rows
before the last input row was written in all three delayed windows. The full
receive remained unavailable at each observation. Sixteen later independent
outputs completed per participant before input zero was supplied. All complete
contractions, received copies, statistics and normalization outputs also passed.
Maximum absolute error was 1.38101313e-7. Each participant used 2436 registered
pages and completed 2356 GPU commands. This establishes partial publication and
composition; it is not a throughput comparison against the earlier implementation.
