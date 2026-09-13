# Pallas collective port: source review

September 13, 2026. Operator correction: implement the Pallas async tensor-parallel
collective structure, use the available shared memory for different outputs in
different destination buffers, and establish correctness from source rather than
transport experiments. This review supersedes any characterization of the current
algebra acceptance program as a completed Pallas collective port.

## What the reference actually allocates

The [Pallas GPU collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
allocates scratch with shape `(axis_size - 1, m_shard, k)`. Each incoming shard
has a distinct destination slice. The guide explicitly connects a smaller scratch
allocation with additional backpressure communication. Its matmul callback forwards
tiles to an indexed remote reference and extends the lifetime of the source tile.
The communication topology is a ring; that does not mean one receive allocation
is repeatedly reused for every incoming shard within the collective.

The [Pallas TPU guide](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#double-buffering)
contrasts double buffering with larger allocations. Two alternating slots require
a bound on sender run-ahead; additional slots allow more overlap. Its all-gather
uses separate output slots for the shards. Pallas still enforces actual data-arrival
and source-lifetime dependencies. More storage removes premature reuse dependencies,
not the need for the requested operand to exist.

## Findings in the current source

1. `rdma/mesh-algebra.m:mesh_algebra_transfer` declares only one local endpoint
   and a numeric binding identity. `rdma/mesh-dataflow.c:mesh_realize` independently
   sorts each participant's local declarations and appends SEND/RECV orders.
   There is no common indexed collective description from which both endpoint
   maps are constructed. Equal binding identity, queue and block expansion are
   caller obligations. The connection exchange checks geometry, not these maps.

2. `rdma/mesh-flow.c:link_post` puts the source address in a SEND work request.
   It contains no remote destination address. The corresponding RECV owns that
   destination. `link_receive` chooses its next row by `next % length`. Thus the
   lowering depends on agreement between complete ordered endpoint sequences;
   successful setup alone cannot establish semantic matching. Different outputs
   already have different receive rows, but their transport identity is positional.

3. `examples/streaming-algebra.c` allocates one instance of each intermediate
   extent and waits for all registered returns before producing the next trial.
   It demonstrates within-invocation partial progress, not an invocation ring or
   Pallas's forwarding pipeline. `mesh_algebra_bind` also builds each numerical
   function with a single work index rather than indexed slot/extent maps.

4. `rdma/mesh.h:mesh_receive_postable` prevents reuse until all readers are done.
   With only one allocation per repeated logical value, this turns allocation
   pressure into a dependency. Making the arena larger does not allocate additional
   value instances or change these function maps. The program must use the memory.

5. `mesh_algebra_scan` submits independent opaque MPS calls and publishes their
   outputs at command completion. It does not implement the reference's callback
   inside the optimized matmul pipeline. Extent-level algebra is useful groundwork;
   it does not establish equivalence to the Pallas async TP implementation.

## The indexing contract to implement

For a configured in-flight depth R, a value instance has coordinates such as
`(invocation_slot, edge, contribution, tile)`. Invocation i uses slot `i % R`.
Every simultaneously live value has distinct canonical registered destination
pages. Different edges include producer outputs, received contributions, reduction
partials and final results; unrelated outputs never acquire identity from arrival
order. Immutable weights may be shared.

For the accepted composition, configure distinct arrays of extents:

```text
producer[slot, group, k_panel]
received[slot, source_peer, group, k_panel]
transformed[slot, group, k_panel]
partial[slot, source_peer_or_reduction_step, group, k_panel]
result[slot, group]
```

Numerical functions name these slices directly. The graph should permit production
into the next slot while consumers read the current slot. Within one finite
collective, allocate separate destinations for all simultaneously live incoming
shards, as the reference does. A slot wraps only after the previous instance's
readers and outstanding source transfers have completed; no finite ring permits
unbounded sender run-ahead. Existing row stamps can express reuse without another
per-job scheduler. R is chosen during configuration from lifetimes and capacity.

The transfer declaration needs both endpoint index maps. Lower each transfer
occurrence from that common declaration to its source pages and its designated
peer destination pages. For SEND/RECV, both participants derive the same per-QP
occurrence order, including invocation slot, edge, contribution, tile and block.
The receiver posts those exact destination pages. This preserves indexed remote
reference semantics despite the substrate lacking remote-write addressing.

For example, a sender order `[A0, B0, A1, B1]` and receiver order
`[A0, A1, B0, B1]` are incompatible even with identical message geometry and every
receive preposted. A setup barrier cannot repair that difference. This is an
illustration of the required proof, not a claim that the measured example used
those mismatched orders. Increasing storage alone also cannot let an anonymous
SEND skip a preceding occurrence on the same QP. Independence must be preserved
in the configured transport order/queue assignment.

## Correction to the previous startup explanation

The previous explanation overstated what the observations established. The receive
WR ID identifies the local posted row; it does not identify the sender's logical
output. Missing receive completions and numerical mismatches did not by themselves
prove which later output landed in which earlier slot, or establish a driver-level
root cause. The setup change accompanied passing measurements. Those observations
do not prove the general endpoint and buffer-reuse contract above.

This review performs no transport experiments and changes no runtime behavior.
The next implementation work is the paired indexed collective and its storage
lifetimes, followed by the numerical pipeline integration. Existing mesh source
and its accumulated documentation are implementation material to audit against
that contract, not authority that the desired Pallas port is already implemented.
