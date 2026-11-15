# Change request: compile unequal shards and ragged values

| Item | Producer and cited mechanism | Result and consumer |
|---|---|---|
| 1. Shard sizes | Placement consumes RaggedShard `local_units` (PyTorch #169320) and rounds sizes by HAP §5.1's least-rounding-error adjustments | Integer counts summing to the dimension, consumed by collective lowering |
| 2. Varying-count collectives | Collective lowering applies MPI 4.1 §§7.5–7.10.2 Gatherv, Scatterv, Allgatherv, Alltoallv and Reduce_scatter | Per-rank counts and displacements on graph transfers |
| 3. Shared counts | Placement derives each rank's count/displacement arrays from the common placement operand, with MPI's matching counts | Consistent endpoint views consumed by routing |
| 4. Fixed local buffer shape | Placement allocates the padded local extent and carries valid sizes/masks as GSPMD's uneven partitioning (§4) | Fixed-shape numerical operands with valid-element semantics |
| 5. Matching partitions | GSPMD sharding propagation (§3.5) aligns dimensions participating in an operation | Compatible operand shardings consumed by the numerical operation |
| 6. Segment sharding ratios | HAP §5's linear program minimizes the sum of communication time at the largest shard and maximum device computation time | One ratio vector per model segment, consumed by placement |
| 7. Size-dependent speed | Lastovetsky et al.'s functional performance model and HAP §3.2 supply speed/work by problem size | Cost functions consumed by item 6; the initial observation uses `x = n / p` (the source algorithm, steps 1–2) |
| 8. Causal attention stripes | Striped Attention §2.2.1 permutes tokens before embedding into rank-modulo stripes, carrying positions and targets through the same permutation | The program retains striped layout and evaluates tiles containing unmasked causal interactions in original token coordinates (§3.1) |
| 9. Resharding | GSPMD §4.5 uses AllGather, AllToAll, CollectivePermute and DynamicSlice | Graph operations carrying item 2's counts and views |
| 10. Halos | GSPMD §4.3 exchanges maximum-size halos with CollectivePermute and selects each rank's extent with DynamicSlice | Window operands consumed by the placed operator |
| 11. Ragged representation | TensorFlow `RaggedTensor` and Arrow List layout represent values with row boundaries | `values` and `row_splits`, consumed by row partitioning and numerical functions |
| 12. Subarray transport | MPI 4.1 §6.1.3 describes a local shard by full sizes, subsizes and starts; ScaLAPACK `descinit` supplies local leading-dimension semantics | Transfer views of the valid shard within its local buffer |
| 13. Row placement | Pınar, Tabak and Aykanat 2008 §§2–4 chain-on-chain partitioning applies LR-PROBE and Nicol's search to prefix sums and device speeds | Ordered separator indices, consumed by placement |
| 14. Capacity-aware ratios | HiGHS variable bounds express `B_j <= u_j` in item 6; Whale Algorithm 1 redistributes a supplied partition using available capacity and computation load | Capacity-feasible placement tuples |
| 15. Default shard sizes | PyTorch DTensor `Shard(dim)` uses `torch.chunk` semantics | Counts including the shorter final shard and trailing empty ranks |

Item 6 uses HAP's fixed distributed program `Q`: `sum(B_j)=1`, `B_j>=0`, and each stage contributes its collective
cost plus the maximum device compute cost. Stage boundaries and model segments follow HAP §§3.2/5.2.
The compiler retains fixed work alongside share-dependent work and passes the resulting coefficients to the solve.

Item 12 selects HAP §2.5.1's padded all-gather or grouped broadcast from their measured cost at the chosen ratios;
the collective-lowering operand carries that selection. Row partitioning uses the given device order and each
row's length as its weight. Ragged offsets remain monotonically increasing; nested ragged dimensions carry nested
offset arrays (TensorFlow RaggedTensor; Arrow Columnar Format).

The shared graph and allocation algorithms consume these explicit counts, layouts, cost terms and placements
through AOT A5/A6/A8. Recompilation reuses retained results whose operands remain the same, following specialization
(Jones, Gomard and Sestoft 1993) and the operator's retained-result instruction.

## Sources

- Zhang et al., *HAP*, EuroSys 2024, arXiv:2401.05965, §§2.4–2.5/3.2/5.
- Xu et al., *GSPMD*, 2021, §§3–4.
- MPI Forum, *MPI 4.1*, §§6.1.3/7.5–7.10.2.
- Lastovetsky, Reddy, Rychkov and Clarke, *Design and implementation of self-adaptable parallel algorithms for
  scientific computing on highly heterogeneous HPC platforms*, 2011, arXiv:1109.3074, functional performance model.
- Brandon et al., *Striped Attention: Faster Ring Attention for Causal Transformers*, 2023, arXiv:2311.09431, §2.2.1.
- Pınar, Tabak and Aykanat, *One-dimensional partitioning for heterogeneous systems: theory and practice*, 2008, §§2–4.
- Jia et al., [Whale](https://www.usenix.org/conference/atc22/presentation/jia-xianyan), USENIX ATC 2022, Algorithm 1.
- [TensorFlow RaggedTensor](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor);
  [Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html);
  [PyTorch RaggedShard](https://github.com/pytorch/pytorch/issues/169320).
