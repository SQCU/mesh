# Composed crossings on torch's stream

A crossing that recurs every step is a persistent exchange ([NCCL.md](../NCCL.md#persistent-exchanges)): planned once,
started each step, its waits and publication compiled into the caller's own kernel. The library runs no kernel of its
own for it. This replaced the fused records of 2026-10-21 and 2026-10-22 (mesh 13688b56 to 292d4516: `ncclMeshFused`,
`ncclMeshFusableRooted`, landing tensors and the six-address bindings the caller passed in its kernel's words).

## The form

**The library** (rdma/).
- `ncclMeshExchangeInit` compiles the exchange every member passes (each rank's part and where it lands) into this
  rank's legs, a member between two unlinked ones relaying ([NCCL.md](../NCCL.md), Persistent exchanges). A slot is one
  lease: this rank's part, a landing area a landing leg and the slot's table. `ncclMeshStart` takes a free slot (every
  kernel bound to it ended, its entries done), else makes one, presents its legs' entries and writes its table: each
  landed part's area, `landed` word and target, the word its own send entries wait on and the value its publication
  stores, and each relayed part's record.
- `ncclMeshBind` binds one record at the kernel's crossing index: the landed start's table and the publishing start's,
  the communicator's failure words, the dispatch's threadgroups and the library's bound. On the classic form it is
  `setBytes`, and the slots' and holdings' buffers are declared with `useResource`, so Metal orders the kernel's pass
  against the passes that name them. No residency set is needed on the classic form.
- The device header (`STEPS`, `ncclMeshMetalPrelude`): `mesh_enter`, `mesh_leave` and the views `mesh_landing`,
  `mesh_own` and `mesh_send`.

**torch-mesh.** `torch_mesh/fused.py` and `_stream.mm`'s `exchange_init`, `exchange_start` and `encode`
([README](README.md#persistent-exchanges-and-kernels-on-torchs-stream)). `exchange.encode` binds the record inside
torch's queue, dispatches, and puts a publishing dispatch under the stream's one commit rule (`MeshStream::published`).

## The chain (examples/)

A step is one start and three dispatches. One more dispatch after the last step lands the last start.

**Stage 0 of step t lands start t − 1.** Every thread of every worker calls `mesh_enter`: thread 0 of each worker
waits for each peer's `landed` (one `volatile coherent(system)` load after a fence a poll, bounded by the library),
then the worker passes a barrier and one system-scope fence. The worker's threads read their own part of start t − 1
(`mesh_own`) and each peer's (`mesh_landing`), add them in float32, own part first and the peers in rank order, round
to float16 once, and use z for x = e + tanh(z); on rank 0 they also write tanh(z), step t − 1's output, into its
staging buffer. `mesh_leave` stores the landed slot's stamp once the last worker has read.

**Stage 2 of step t publishes start t.** The `Summand` epilogue stores the part through `mesh_send` (system-coherent),
each worker its own columns, into the slot's own pages: the bridge sends those pages, nothing copied. `mesh_leave`:
each worker fences and counts its arrival, and the last to arrive stores the slot's `published` (the word its send
entries wait on) after one system-scope fence, publishes each part the rank relays once it has landed, counts the
kernel's end in the slot's table, and zeroes the count of arrivals for the slot's next use. No worker waits for another to publish.

Start t + 1 lands in other pages than start t whenever start t's are still held: no landing tensor is reused while a
kernel may read it, and no send part while the NIC may read it. The kernel carries no guard and no binding of its own.

**chain-service.py.** The input ring is a region tensor a block (each a lease, so a block starts on a page without the
service knowing the page), filled and broadcast a block ahead. The plain all-reduce timing (`crossing_microseconds`)
keeps its sum ring, a region tensor a slot.

## Readings

Nothing has run in this form. The readings of the fused form on main, for comparison when it runs: the chain's float64
check 1.37e-3 at step 0 and 1.19e-3 at step 3,584; a steady step of 35.4 µs; the plain crossing 17.2 µs, about 12 µs
bare. The harness logs aggregate rows/s, steps/s per stream, the host and paced times and the parts tuple, and
`ncclMeshTimes` gives the crossings' groups' times.
