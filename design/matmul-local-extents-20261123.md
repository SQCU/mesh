# Matrix products on unequal column shards, 23 November 2026

**A:** Placement rounded the measured work shares to exact integer counts, then widened each rank’s matrix operand to the largest count. An 84/44 split of 128 output columns issued two 84-column matrix products. The gather used only the 128 assigned columns.

**B:** Each rank passes views of its assigned rows, columns, or contraction range to its own matrix call. The same shape flows through a pointwise operation or transpose. A downstream operation that consumes a padded layout obtains it at that boundary. MPI counts and displacements still describe the valid ranges. This distinguishes a fixed storage extent from the iteration domain of a local contraction. [Unequal shard counts, fixed storage and valid subarrays](unequal-shards-change-request.md)

The threads-and-queues lowerer executed a 19 × 17 by 17 × 13 product and a two-product feed-forward graph with three ranks. The runs exercised unequal contiguous counts in both rank orders, a zero-sized shard, and striped partitions of rows, columns, or the reduction dimension. The sums of the local matrix multiply-add counts equaled the original graphs’ counts. Both input batches under each placement matched eager PyTorch.
