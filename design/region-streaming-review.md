# Producer and consumer streaming: source and literature review

Reviewed September 13, 2026, starting from `99195b3`.

## Literature and the implemented mechanism

The JAX authors' [software pipelining guide](https://docs.jax.dev/en/latest/pallas/pipelining.html#pipelining-basics)
separates slice transfer, slice computation, and output transfer. Independent
iterations overlap; multiple buffers remove false dependencies caused by reuse.
Its completion waits belong immediately before dependent accesses, not before
independent work. This supports mesh's region presence and last-reader masks.

The [collective matmul](https://docs.jax.dev/en/latest/pallas/gpu/collective_matmul.html)
uses a callback inside the computation pipeline to forward operand tiles, with
completion handling for sends and buffer reuse. It also includes a coarser
receive wait. Copying that example literally would not prove the stronger mesh
requirement of consuming every independently available region.

The [nested TPU pipelines](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#nested-remote-and-local-dma-pipelines)
separate remote transfer and local numerical blocking. Mesh likewise must not
confuse an allocation's extent with an operation's dependency extent.

Blelloch's [Prefix Sums and Their Applications](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html)
provides associative scan/reduction composition. It does not establish that the
current mesh reduction tree implements an arrival-selected masked accumulator.

## Findings

### Fixed: generic numerical functions required whole allocations

`mesh_algebra_function` called `full_output` and assigned all extent rows to each
output. `BlockSpec` could select an allocated block but could not express an
output subregion. A custom producer therefore published a large allocation only
on whole-allocation completion. A custom consumer using that same reference
required the whole allocation to arrive.

The function now accepts compact disjoint output regions aligned to the existing
publication quantum. `BlockSpec.region_map` selects input and output subregions
during setup. Output collision checks compare actual row ranges, including input
read ranges, rather than rejecting every reference to the same allocation.
Completion publishes only those output rows. Existing whole-extent transfer
bindings already enumerate their transfer blocks and observe those publications.
No transfer wrapper, second scheduler, or runtime allocation was added.

A trailing region may include the allocation's unused padding. An interior region
must end on a publication boundary so publishing it cannot expose unwritten
payload. Strided inputs remain valid; output regions must own contiguous storage.

### Open: a fixed reduction tree is not an arrival-selected masked sum

`Program.reduce_sum` and `mesh_algebra_contract` create adjacent-pair ADD
functions. `mesh_issue` requires every input of one such function. If contributions
1 and 2 arrive while 0 and 3 are absent, both contributions can be computed, but
neither predetermined pair can add yet. There is no implementation that selects
those two ready contributions into an accumulating sum.

The distinction is substantive. A readiness mask selects existing contributions;
it does not make absent data readable or make a partial sum final. For disjoint
sets of contributions U and V, the numerical composition is

    (sum(U), mask(U)) + (sum(V), mask(V))
      = (sum(U) + sum(V), mask(U) OR mask(V)).

Their masks must be disjoint to avoid counting a contribution twice. Finality is
coverage of the required mask. Existing PRESENT/READ bits already own presence
and consumption; a future lowering must use that ownership rather than inventing
a second task scheduler or treating an absent contribution as permanently zero.
This review does not claim that primitive is implemented.

### Open: contraction and row reduction depend on the configured input partition

`bind_part` narrows output rows, but a contraction still names all K entries in
its supplied operand view; a row sum names all columns of its supplied view.
`Tensor.__matmul__` and `Tensor.sum` split those dimensions only across existing
blocks. A tensor created with the default whole-shape block can therefore retain
a whole-K/whole-row receive dependency. This is not established as unavoidable:
finer contributions can be computed and composed before that operand completes.
The library does not yet automatically realize that finer input partition.

### Verified connections and their limits

`dependencies` maps declared input views to pages. `mesh_issue` checks only those
maps and output reuse masks. `complete_part` calls `mesh_complete`, which marks
outputs PRESENT and configured send rows HOT. The independent bridge reads those
canonical registered pages; receive completion makes destination rows present.
A consumer whose declared regions are complete can run while unrelated regions
remain absent. Physical SEND completion retains source storage until NIC reads
finish. These mechanisms enable overlap but do not erase the open dependencies
above. The earlier claim that the full streaming objective was complete was too
broad.

Compilation and source review validate the region API change. No synthetic
numerical client or timing result is used as proof of overlap.
