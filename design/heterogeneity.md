# Heterogeneous nodes and links

Orientation for anyone, developer or agent, who splits work over nodes of different speeds or
moves it over links of different costs: `rdma/torch-mesh/torch_mesh/partition.py` (the partition
operand), `rdma/allocate.py` (the shares), `rdma/mesh-plan.h` and `mesh-collective.c` (the compiled
collectives) and the programs that use them. The operator:

> "it'll take a little bit of orientation for developers (agents as well) to handle heterogenous
> links ontologically (since this is so different) and without lots of clumsy erratic
> overcorrections (this is mostly requirements in the literature, yo)."

None of it is new. Stock PyTorch splits a sharded dimension equally (`Shard` follows `torch.chunk`),
a uniform prior over the ranks. Splits in proportion to device speed are heterogeneous ScaLAPACK's
[Beaumont et al. 2001], AccPar's partition ratios [Song et al. 2020], HAP's sharding ratios
[Zhang et al. 2024] and veScale's `RaggedShard(local_units)` [pytorch#169320]. The collectives they
need are MPI's v-collectives [MPI 4.1 §6.5-6.10].

## 1. Vocabulary

- **Stretch**: a rank's own work between two consecutive collectives, at the call's shape and dtype, timed on its own
  clock or torch's MPS stream (`torch_mesh/evidence.py`; metal-microbench `mesh_rank.m`). A collective's wait is in no
  stretch (R2). Stretches are the evidence the parts move on (R3). Node rates per operator class (`proj=u,v`,
  `attn=u,v`), which the rate derivation read, are read by no code since fa36bea6.
- **Link**: the cost of one direction of one cable between a pair of nodes, α (µs) per message plus β (ns) per byte.
  The bridges' map joins two nodes by one link for each cable that both ends' records name, at the lesser of the two
  ends' bandwidths for it (mesh-plan.h `mesh_map`: β 8e9 over that bandwidth, mesh-dataflow.c `mesh_joined`).
- **Parts**: a mesh dimension's integers p_r ≥ 1, one per coordinate, of P = Σp units.
- **Share**: rank r's part of an extent N, N·p_r/P.
- **Grain**: N/P, the unit a share is counted in (a head, an expert, a tile). A program chooses the
  grain by choosing P.
- **Counts**: a collective's elements for each rank, a whole number of grains each; every rank holds
  the same list of counts (R4).
- **Finish time**: when a rank's segment ends: its compute at its share plus the collectives it
  waits for. The step ends at the largest finish time.

## 2. Two facts, two mechanisms

| Fact | Written in | Read by | Decides |
|---|---|---|---|
| Measured stretches | each rank's own clock over its work between two collectives (`torch_mesh/evidence.py`; metal-microbench `mesh_rank.m`) | `allocate.Balancer` (`partition.rebalance`, section 4; metal-microbench `tools/mesh/rates.py`). The rate derivation that read node rates, `allocate.equal_finish`, was deleted in fa36bea6 (2026-10-17), and metal-microbench `programs.py derive` in a275ddf5 (2026-10-20) | the parts: what each node computes |
| Links | the bridges' map, one link with its cost between two nodes whose bridges reach each other, and the routing table (`mesh_map`; a communicator's map holds its members alone, `nccl-mesh.c` `map_of`: a route crosses members, one with no work of a call sending on what lands) | the compiled collectives (`mesh_compile` along shortest-path trees balanced over the links, or a configured form's packed trees cut into pipelined chunks, `ncclMeshCommForms`, or a program's own, `ncclMeshCommInitRank`, torch-mesh `Options`) | the collective's form, its trees and its chunks: how operands travel |

- Measured stretches move the parts. The map decides the algorithm.
- A stretch excludes the collectives' waits (R2), so a link's cost does not read as a node's time, and a node's time
  selects no algorithm.
- R7's coupling, the collectives a share waits for priced at the largest share [HAP 2024, §2.4], was the rate
  derivation's, deleted in fa36bea6. The Balancer reaches it only where a stretch carries its crossing's latency: it
  then weighs the program's longest path (section 4).
- Shifting shares cannot route around a slow link. In a ring, every block crosses n−1 of the n
  links [Patarasuk & Yuan 2009], so a slow link carries every block but one whatever the shares.
  The shares change only how many bytes it carries (N minus the destination's share).

## 3. What the literature specifies

| | What the code does | Source | The overcorrection it avoids |
|---|---|---|---|
| R1 | No operand means stock PyTorch, bit for bit. Equal parts also mean stock, including `torch.chunk`'s ceil sizes (10 over 4 is 3,3,3,1, not largest remainder's 3,3,2,2). | [torch.chunk]; the regular operation is the irregular one's special case and no slower than it [Träff, Gropp & Thakur 2010; Hunold & Carpen-Amarie 2017, GL4] | a second "uniform" rule that silently changes programs that never set an operand |
| R2 | What each node computes follows from the ranks' measured stretches, each its own work between two collectives, excluding the collective's wait (`torch_mesh/evidence.py`). How operands travel follows from the link map. No rate or price couples the two (fa36bea6). | metal-microbench `docs/principles.md` §4; link-aware collectives [MagPIe 1999; BlueConnect 2019; Blink 2020]; design/rubrics.md D6 | a node's time measured with its collectives inside it, so a slow link reads as a slow node; reshaping collectives to "fix" a slow node |
| R3 | Evidence is per rank, per decision and per stretch, from the mesh's own calls at the program's shape and dtype: `allocate.Balancer` keeps each (decision, stretch, rank)'s measured points as a functional model and moves the parts between calls. The first parts are the program's own (`partition.attach`), and no price or table sets them. Every part whose bound holds a grain keeps at least one (`Balancer._floor`), so a loaded member gets less work, not none. | functional performance model, partially estimated [Lastovetsky & Reddy 2007; DFPA 2011]; design/rubrics.md D5, the operator: "the *second entire computer* is a resource" (10-17), and D6: "cost model inference inside of leaf closures" ≺ "iterative estimators which update between function calls" (10-05) | one speed per node applied to every operator; a price from a table, another class, another dtype or a solo-node measurement setting a share; a loaded member benched at 0% |
| R4 | Counts are derived once per configuration, from shared configuration, by one deterministic procedure, so they are identical on every rank. | counts agree across ranks: recvcounts are the same on all members [MPI 4.1 §6.7, §6.10.2]; the distribution follows from a declared performance model [HeteroMPI 2006] | counts from each rank's own timing: ranks that disagree hang or corrupt |
| R5 | Integer rounding minimises the largest finish time: each grain goes to the rank that finishes earliest with it. | [Beaumont et al. 2001, Alg. 3.1]; greedy allocation is optimal for nondecreasing costs [Ibaraki & Katoh 1988] | largest remainder rounding the slow rank up by a whole grain |
| R6 | Tensors that meet in one operation split their shared extent alike, and every rank can invert its own split from its local extent and the capacities (`from_local`: global = local·P/c_r, R14). | the first GEMM's column split is the second's row split [Megatron-LM 2019, §3]; the literature's ratios are per tensor, `RaggedShard(local_units)` on each placement [pytorch#169320], and per layer [HAP 2024, §2.4] | rounding each extent separately, or splitting an extent the parts do not divide by `torch.chunk`: views and the inverse break, and ranks infer different global shapes |
| R7 | The collectives a share waits for cost most at the largest share. No code prices them ahead of a call since fa36bea6 deleted the rate derivation; the Balancer's longest path includes a crossing's latency where a stretch carries it (section 4). | communication time depends on the largest shard; even ratios win "when communication is the bottleneck", and since layers differ in computation-to-communication ratio, the optimal ratios may vary for each layer [HAP 2024, §2.4]; the ring's irregular all-gather is dominated by the largest block [Träff et al. 2010] | compute-proportional shares on communication-bound work |
| R8 | Equal counts run the regular collective. Unequal counts run the v-collective with each block at its own size, never padded to the largest. | Gatherv, Scatterv, Allgatherv, Alltoallv, Reduce_scatter [MPI 4.1 §6.5-6.10]; regular ⪯ irregular [Hunold & Carpen-Amarie 2017, GL4, GL8, GL12, GL18, GL22]; pad or per-rank broadcast [HAP 2024, §2.5.1; pytorch#198344] | padding's wasted bytes and region memory; the v-form's overhead on even splits |
| R9 | What a node can hold is a hard upper bound on its share. A share that exceeds it is clamped and the rest is solved again. | memory-bounded load balance [Whale 2022; Metis 2024]; GPU memory limits the batch [LB-BSP 2020]; bounded variables held at their bound and the rest re-solved [Bitran & Hax 1981] | a share the fast node cannot hold, which pages and makes the fast node slow |
| R10 | Nonlinear work is balanced on its own cost function (`allocate.min_max` takes any nondecreasing cost, and `allocate.Balancer` balances on measured costs; `equal_finish`, on affine costs a + b·s, was deleted in fa36bea6). Causal attention's work depends on where a share lies in the sequence; head-tail pairing makes it depend on the share alone. | causal ring attention's imbalance [Striped Attention 2023]; [Lastovetsky & Reddy 2007] | linear weights on quadratic work |
| R11 | Parts change with the configuration, not per step. An online loop follows its sources: it predicts robustly and reduces its step once it oscillates, stops at a relative accuracy, and rebalances only when the gain exceeds the cost. `allocate.Balancer` has all of it: models over a window of calls (`window`, 3 by default), half steps after a reversal, a stop at its resolution (twice the measured sum's relative standard error, at least `RESOLUTION`, 1 %), and a move only where the measured models promise more than that resolution (metal-microbench `tools/mesh/rates.py` runs it after every committed call). `tools/mesh/balance_sim.py`, deleted with the planner's cost model in f2b75b7e on 2026-10-17, validated `allocate.dfpa` on simulated M1-M12 and counterfactual nodes; 8e33f8a9 replaced `dfpa` with the Balancer, and 4fff4754 deleted `allocate.Allocator`. | static distributions avoid redistribution and control overhead [Beaumont et al. 2001, §2.2]; prediction robust to non-deterministic perturbation "to avoid over-reaction or oscillation" [LB-BSP 2020, §3.2.1], an observation window and a smaller step once oscillation is detected [LB-BSP 2020, §3.3.2]; stop at a relative accuracy ε [DFPA 2011]; invoke the balancer when the gain exceeds its cost [Meta-Balancer 2012] | ping-pong after one noisy step; re-sharding weights because of jitter |
| R12 | Overlap hides a collective behind independent work, but it shortens a rank's own finish, not its wait for a slower rank. Decomposed pieces follow the counts, and k pieces cost k·α. | overlap by decomposition [Wang et al. 2023; Domino 2024; FLUX 2024]; balance first because the step ends at the largest finish time (§1); k·α in the α-β model [Hockney 1994] | using overlap to cure an imbalance |
| R13 | Anything the partition cannot express falls back to stock's own fallback, or raises the same error on every rank. Nothing silently computes a wrong shape or value. | — | silent numerics, such as a mean of unequal shards averaged as if the shards were equal |
| R14 | Parts change without shapes changing. Each rank's local buffer is shaped to the largest share the bounds allow it (its capacity, stated with the parts); the parts are a tensor operand read when the ops run. Padding holds the identity of its next consumer: sources write it (weights zero-padded at load, collectives the identity past the valid extent), and before any op that reduces or contracts over a partitioned dimension the shard is re-masked to that op's identity (0 for sums and contractions, -inf for max and masked softmax); attention masks keys past the valid extents; count-dependent ops divide masked sums by the valid count. Kernels are stock: they compute garbage in padding that nothing reads unmasked. | uneven partitioning: pad to the shards' size and mask the padding to the identity of the next operation wherever it could leak into valid results [GSPMD 2021, §3.3] | shapes that follow the parts, so every change of parts recompiles, re-records and reallocates a program |

## 4. The operand in use

The module docstring (`torch_mesh/partition.py`) is the API. This section is only its shape.

**Use.** `torch_mesh.partition` installs nothing into torch (metal-microbench docs/acceptance-20261014.md A53): a
program uses its functions where a partitioned dimension is meant, and torch's own run as torch runs them.
- On MPS the mesh is formed from the process groups, `DeviceMesh.from_group(dist.group.WORLD, "mps",
  mesh_dim_names=("tp",))` (torch 2.14's `init_device_mesh("mps")` needs device calls its MPS module lacks).
- The partition is a placement, `RaggedShard(dim, key)` (a subclass of torch's `Shard`, which the functions below take
  for it on an attached dimension), and a DTensor holding one is a `RaggedTensor`, whose own dispatch keeps it
  ragged through every op, re-masks the padding before each sum, max or min over a ragged dimension and gathers the
  dimension whole where an op needs it so (the module docstring lists every torch entry point it uses).
- `partition.distribute_tensor`, `partition.from_local` and `partition.redistribute`, `partition.ColwiseParallel`,
  `RowwiseParallel` and `SequenceParallel`, and context parallelism's `partition.context_parallel_buffers`,
  `ring_attention` and `context_parallel_unshard` are torch's with the attached dimension's `RaggedShard`.

**Operand.**
- `partition.attach(mesh, tp=(capacity, parts))`, before the mesh is first used, the same on every
  rank; a sub-mesh keeps its root's. `partition.write(mesh, tp=parts)` changes the parts between calls.
- `partition.distribute_tensor(x, mesh, [Shard(d)], src_data_rank=None).to_local()` and `partition.split(mesh, "tp",
  N)` give a program its own splits; `x.chunk(world)` gives equal ones.
- `partition.from_local` without `shape=` means "a capacity block" (global = local·P/c_r).
- Every extent split along a partitioned dimension is a multiple of P; any other raises
  `ValueError`, on every rank alike. No mesh dimension of more than one coordinate precedes a
  partitioned one (attach raises otherwise), so no outer split reaches a partitioned dimension first
  and the extent it splits is the global one: DTensor splits a tensor dimension sharded on several
  mesh dimensions outer first, and an outer split's shares can differ by coordinate, so ranks would
  disagree about P dividing them. An inner dimension is partitioned on a mesh ordered with it first.

**First parts.** The program states them: `partition.attach(mesh, tp=(capacity, parts))` takes the capacities and
parts as operands, the same on every rank (`partition.py` `attach`), and metal-microbench `tools/torch_parallel/tp.py`
takes them as `PARTITION`. No code in this repository derives them from node rates. Until 2026-10-17 a
derivation did: each rank's time for c units was a_r + b_r·c from its node's rates per operator class, the collectives
on the dimension's critical path were added at the largest share by `mesh.program_time`, and
`allocate.equal_finish(P, 1, low, high, a, b)` gave the integer parts. cc8a6f8e deleted `mesh_program_time` with
the planner's timing, and fa36bea6 deleted `equal_finish` with the fitted models. `allocate.min_max` remains: it
divides units among ranks given nondecreasing time functions, each grain to the rank finishing earliest with it (R5).

Links and their costs are the bridges' map. They set the collectives' form (section 2), not the parts.

**Parts from the run's evidence (R3, R10, R11).** `allocate.Balancer` balances any program of collectives: its
decisions are partitioned dimensions, its stretches each rank's own work between two consecutive collectives
(never a collective's wait, R2), and each decision's parts minimise its stretches' part of the call's time, the sum
over stretches of the slowest rank's, or, where each stretch carries its place in its program and its crossing's
latency, the program's longest path (a stretch starts on a rank once its own previous one has ended and every other
rank's has ended and crossed: where the slower rank alternates the call is the cross cycle, which the sum misses)
(DFPA on the measured points; a trust region never less than a grain past the measured parts). `partition.rebalance(mesh, name, every)`, called
once a step on every rank, watches the dimension's group (`torch_mesh/evidence.py`: a stretch a collective, timed on
torch's MPS stream or the host clock); every `every` steps it gathers the ranks' stretches to all of them and runs
one Balancer step, identically on every rank (R4). A prepared call's stretches are its ranks' own: each rank times,
at every crossing it issues on the host path, its work since its previous crossing ended and the crossing's own time,
on its own clock (metal-microbench `mesh_rank.m`). Its parts stand once the evidence promises no gain past its resolution, so they change with the
configuration and not with jitter (R11). A rank's work follows its capacity (R14: stock kernels compute the
padding), so parts that move past the capacities are attached again at them; that is the one cost R11 weighs.
metal-microbench's calls run the same Balancer on those stretches (`tools/mesh/rates.py`), its parts and the
evidence they moved on kept in the mesh's store (`rdma/store.py` 'configurations').

**What stays uniform.**
- No operand, or equal parts: stock PyTorch.
- The strategy costs (`MeshTopoInfo`, `redistribute_cost`). They choose among collectives, not
  splits.

**Limits of this module.**
- One set of parts per mesh dimension. This is a limit of the module, not a requirement: the parts
  ride on `DeviceMesh.size()`, which Shard's size function reads without the tensor. The
  literature's parts are per tensor and per layer (R6, R7). The design's optional per-tensor
  override is not built.
- The grain is the only over-decomposition. Over-decomposition into many more objects than
  processors, placed by the run-time system [Charm++ 1993], is the fallback alternative to derived
  parts and is not built here.
- What cannot take uneven shards falls back or raises; the docstring lists these cases.

## References

- Beaumont, O., Boudet, V., Petitet, A., Rastello, F. & Robert, Y. (2001). A proposal for a heterogeneous cluster ScaLAPACK (dense linear solvers). IEEE Trans. Computers 50(10). https://doi.org/10.1109/12.956091
- Bitran, G. & Hax, A. (1981). Disaggregation and resource allocation using convex knapsack problems with bounded variables. Management Science 27(4). https://doi.org/10.1287/mnsc.27.4.431
- Blink: Wang, G. et al. (2020). Blink: fast and generic collectives for distributed ML. MLSys. https://arxiv.org/abs/1910.04940
- BlueConnect: Cho, M., Finkler, U., Kung, D. & Hunter, H. (2019). BlueConnect: decomposing all-reduce for deep learning on heterogeneous network hierarchy. SysML. https://proceedings.mlsys.org/paper_files/paper/2019/hash/0c8abcf158ed12d0dd94480681186fda-Abstract.html
- Charm++: Kalé, L. & Krishnan, S. (1993). CHARM++: a portable concurrent object oriented system based on C++. OOPSLA. https://doi.org/10.1145/165854.165874
- DFPA: Lastovetsky, A., Reddy, R., Rychkov, V. & Clarke, D. (2011). Design and implementation of self-adaptable parallel algorithms for scientific computing on highly heterogeneous HPC platforms. https://arxiv.org/abs/1109.3074
- Domino: Wang, G. et al. (2024). Domino: eliminating communication in LLM training via generic tensor slicing and overlapping. https://arxiv.org/abs/2409.15241
- FLUX: Chang, L.-W. et al. (2024). FLUX: fast software-based communication overlap on GPUs through kernel fusion. https://arxiv.org/abs/2406.06858
- HAP: Zhang, S. et al. (2024). HAP: SPMD DNN training on heterogeneous GPU clusters with automated program synthesis. EuroSys. https://arxiv.org/abs/2401.05965
- HeteroMPI: Lastovetsky, A. & Reddy, R. (2006). HeteroMPI: towards a message-passing library for heterogeneous networks of computers. JPDC 66(2). https://doi.org/10.1016/j.jpdc.2005.08.002
- Hockney, R. W. (1994). The communication challenge for MPP: Intel Paragon and Meiko CS-2. Parallel Computing 20(3). https://doi.org/10.1016/S0167-8191(06)80021-9
- GSPMD: Xu, Y. et al. (2021). GSPMD: general and scalable parallelization for ML computation graphs. https://arxiv.org/abs/2105.04663
- Hunold, S. & Carpen-Amarie, A. (2017). Tuning MPI collectives by verifying performance guidelines. https://arxiv.org/abs/1707.09965
- Ibaraki, T. & Katoh, N. (1988). Resource Allocation Problems: Algorithmic Approaches. MIT Press. https://dl.acm.org/doi/book/10.5555/49354
- Lastovetsky, A. & Reddy, R. (2007). Data partitioning with a functional performance model of heterogeneous processors. IJHPCA 21(1). https://doi.org/10.1177/1094342006074864
- LB-BSP: Chen, C., Weng, Q., Wang, W., Li, B. & Li, B. (2020). Semi-dynamic load balancing: efficient distributed learning in non-dedicated environments. SoCC. https://doi.org/10.1145/3419111.3421299
- MagPIe: Kielmann, T. et al. (1999). MagPIe: MPI's collective communication operations for clustered wide area systems. PPoPP. https://doi.org/10.1145/301104.301116
- Megatron-LM: Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J. & Catanzaro, B. (2019). Megatron-LM: training multi-billion parameter language models using model parallelism. https://arxiv.org/abs/1909.08053
- Meta-Balancer: Menon, H., Jain, N., Zheng, G. & Kalé, L. (2012). Automated load balancing invocation based on application characteristics. IEEE Cluster. https://doi.org/10.1109/CLUSTER.2012.61
- Metis: Um, T. et al. (2024). Metis: fast automatic distributed training on heterogeneous GPUs. USENIX ATC. https://www.usenix.org/conference/atc24/presentation/um
- MPI Forum (2023). MPI: A Message-Passing Interface Standard, version 4.1, chapter 6. https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf
- Patarasuk, P. & Yuan, X. (2009). Bandwidth optimal all-reduce algorithms for clusters of workstations. JPDC 69(2). https://doi.org/10.1016/j.jpdc.2008.09.002
- pytorch#169320: [DTensor] A new placement: RaggedShard. https://github.com/pytorch/pytorch/issues/169320
- pytorch#198344: [c10d][NCCL] Uneven all_gather_into_tensor / reduce_scatter_tensor. https://github.com/pytorch/pytorch/issues/198344
- Song, L. et al. (2020). AccPar: tensor partitioning for heterogeneous deep learning accelerators. HPCA. https://doi.org/10.1109/HPCA47549.2020.00036
- Striped Attention: Brandon, W. et al. (2023). Striped attention: faster ring attention for causal transformers. https://arxiv.org/abs/2311.09431
- torch.chunk. https://docs.pytorch.org/docs/stable/generated/torch.chunk.html
- Träff, J. L., Gropp, W. & Thakur, R. (2010). Self-consistent MPI performance guidelines. IEEE TPDS 21(5). https://doi.org/10.1109/TPDS.2009.120
- Träff, J. L., Ripke, A., Siebert, C., Balaji, P., Thakur, R. & Gropp, W. (2010). A pipelined algorithm for large, irregular all-gather problems. IJHPCA 24(1). https://doi.org/10.1177/1094342009359013
- Whale: Jia, X. et al. (2022). Whale: efficient giant model training over heterogeneous GPUs. USENIX ATC. https://www.usenix.org/conference/atc22/presentation/jia-xianyan
- Wang, S. et al. (2023). Overlap communication with dependent computation via decomposition in large deep learning models. ASPLOS. https://doi.org/10.1145/3567955.3567959
