# Streaming algebra with indexed destinations

The substrate represents values as independently usable extents in canonical
registered shared memory. Numerical coordinates, readiness extents and transport
blocks are separate. DNN modules compose the algebra operations; the transport
and algebra headers contain no model definition.

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
Every participant numbers every occurrence before projecting its local work.
The source peer contributes a SEND binding and the destination peer contributes
a RECV binding with the same occurrence identity. A same-peer transfer becomes
a local affine copy. No tensor payload is packed into another transport store.

All participants execute the same ordered copy declarations with the same global
parameters. Local numerical graphs and physical addresses need not be identical.
The old mesh_algebra_transfer entry point remains available for manually managed
bindings; a paired program uses mesh_algebra_copy throughout so its occurrence
identities have one owner. It does not mix independent manual identities into
that same transfer schedule.

## Storage and execution

Create tensors and bind numerical functions before mesh_algebra_realize. Configuration
allocates the registered pages, zero-copy Metal aliases, layouts, specializations
and matrix bindings. Numerical invocation scans the configured functions with
mesh_issue, submits their GPU commands, and publishes with mesh_complete.
No allocation of operand storage or choice of backend occurs during invocation.

Affine, weighted addition, multiplication, tanh, exp, row sum, reciprocal square
root and MPS contraction are available. Views can slice, transpose and broadcast.
Every output covers its complete independently allocated extent. An input slice
retains its containing extent's readiness. Partial contractions have separate
outputs and explicit additions fix their association.

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

Its transfer occurrences are uniquely numbered as follows, with extent
i = group + 2 * k_panel:

| Occurrence | Source | Destination | Queue |
|---|---|---|---|
| 10 * slot + 2 * i + peer | peer's producer[slot, i] | other peer's received[slot, i] | group |
| 10 * slot + 8 + peer | peer's result[slot, peer] | other peer's gathered[slot, peer] | peer |

Both peers enumerate all ten occurrences per slot. Projection removes only
occurrences not locally owned. mesh_realize orders each queue's blocks by
occurrence identity and block index, so each source SEND sequence equals the
corresponding destination RECV sequence. The local row/page numbers can differ.
This establishes destination identity from source, not from payload arrival time.
The live slots use disjoint allocations. Reuse is subject to those pages' actual
reader lifetimes; the ring of work-request metadata is not the value-buffer ring.

For each window the caller withholds input zero of its first invocation, while
producing all other inputs into their separate slots. It requires group one's
contraction to complete in every invocation in the window before supplying the
withheld input. With depth eight, seven later invocations must produce that output
while the first invocation still lacks input zero. No outputs have been consumed
at that observation point. Afterward the host consumes and refills one slot at a
time, while the same static scan runs all eligible numerical functions.

Queue zero's FIFO can still block later group-zero transfers behind the missing
first input. Queue one has a separate configured order and carries the independent
work. More memory removes destination-reuse dependencies; it does not make an
anonymous SEND randomly address a later slot on the same queue.

## Build and acceptance

Build `make -C rdma .build/streaming-algebra`. Run
`rdma/.build/streaming-algebra 0 32 128 8` and
`rdma/.build/streaming-algebra 1 32 128 8` on the two peers. Arguments are rank,
invocation count, rows and depth. A 4096-page arena with 4-page blocks and two
queues accommodates this example. Shape 257 also exercises extents spanning
multiple transport blocks. The existing executable is the acceptance path.

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

MPS contractions publish at submitted extent boundaries. This implements indexed
storage and async extent composition, not Pallas's inner-matmul forwarding callback.
Intra-dispatch publication needs a numerical backend exposing those boundaries.
FP16 is supported by the interface but has not been covered by this example.
