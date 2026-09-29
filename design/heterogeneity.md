# Heterogeneous nodes and links

Orientation for anyone, developer or agent, who splits work over nodes of different speeds or
moves it over links of different costs: `rdma/torch-mesh/torch_mesh/partition.py` (the partition
operand), `rdma/allocate.py` (the derivation), `rdma/mesh-collective.c` (the planner) and the
programs that use them. The operator:

> "it'll take a little bit of orientation for developers (agents as well) to handle heterogenous
> links ontologically (since this is so different) and without lots of clumsy erratic
> overcorrections (this is mostly requirements in the literature, yo)."

None of it is new. Stock PyTorch splits a sharded dimension equally (`Shard` follows `torch.chunk`),
a uniform prior over the ranks. Splits in proportion to device speed are heterogeneous ScaLAPACK's
[Beaumont et al. 2001], AccPar's partition ratios [Song et al. 2020], HAP's sharding ratios
[Zhang et al. 2024] and veScale's `RaggedShard(local_units)` [pytorch#169320]. The collectives they
need are MPI's v-collectives [MPI 4.1 §6.5-6.10].

## 1. Vocabulary

- **Rate**: a node's time for one operator class at the call's shape, `a + u·bytes + v·ops`. It is
  written on the node's line of the link map. A node has one rate per class, never one speed.
- **Link**: the cost of one direction of one pair of nodes, α (µs) per message plus β (ns) per byte.
  It is written on a link line of the link map.
- **Parts**: a mesh dimension's integers p_r ≥ 1, one per coordinate, of P = Σp units.
- **Share**: rank r's part of an extent N, N·p_r/P.
- **Grain**: N/P, the unit a share is counted in (a head, an expert, a tile). A program chooses the
  grain by choosing P.
- **Count**: a collective's elements per rank. It is a whole number of grains and is the same on
  every rank.
- **Finish time**: when a rank's segment ends: its compute at its share plus the collectives it
  waits for. The step ends at the largest finish time.

## 2. Two facts, two mechanisms

| Fact | Written in | Read by | Decides |
|---|---|---|---|
| Node rates | the link map's node lines | the derivation (`allocate.min_max`) | the parts: what each node computes |
| Links | the link map's link lines | the planner (`mesh_collective_choose`; libnccl-mesh through `MESH_LINKS`) | direct, ring or tree: how operands travel |

- A slow node is answered by its share. A slow link is answered by the algorithm.
- The link map enters the derivation only as the price of the collectives a share waits for (R7). It
  never becomes a node's rate, and a node's rate never selects an algorithm.
- Shifting shares cannot route around a slow link. In a ring, every block crosses n−1 of the n
  links [Patarasuk & Yuan 2009], so a slow link carries every block but one whatever the shares.

## 3. Requirements

| | Requirement | Source | Overcorrection it prevents |
|---|---|---|---|
| R1 | No operand means stock PyTorch, bit for bit. Equal parts also mean stock, including `torch.chunk`'s ceil sizes (10 over 4 is 3,3,3,1, not largest remainder's 3,3,2,2). | [torch.chunk]; the regular operation is the irregular one's special case and no slower than it [Träff, Gropp & Thakur 2010; Hunold & Carpen-Amarie 2017, GL4] | a second "uniform" rule that silently changes programs that never set an operand |
| R2 | What each node computes follows from the nodes' rates. How operands travel follows from the link map. Neither is adjusted to fix the other. | metal-microbench `docs/principles.md` §4; link-aware collectives [MagPIe 1999; BlueConnect 2019; Blink 2020] | skewing compute shares to "fix" a slow link; reshaping collectives to "fix" a slow node |
| R3 | Rates are per node **and per operator class**, taken at the call's shape (bytes and operations), from a functional model. | functional performance model [Lastovetsky & Reddy 2007]; [Roofline 2009]; per-device memory bandwidth and FLOPS in the cost model [HexGen 2024, App. B]; ratios per layer [HAP 2024, §2.4] | one speed per node applied to every operator; bandwidth-bound (decode) rates applied to compute-bound work |
| R4 | Counts are derived once per configuration, from shared configuration, by one deterministic procedure, so they are identical on every rank. | counts agree across ranks: recvcounts are the same on all members [MPI 4.1 §6.7, §6.10.2]; the distribution follows from a declared performance model [HeteroMPI 2006] | counts from each rank's own timing: ranks that disagree hang or corrupt |
| R5 | Integer rounding minimises the largest finish time: each grain goes to the rank that finishes earliest with it. | [Beaumont et al. 2001, Alg. 3.1]; greedy allocation is optimal for nondecreasing costs [Ibaraki & Katoh 1988] | largest remainder rounding the slow rank up by a whole grain |
| R6 | A mesh dimension has one set of parts. P divides every extent split by it. The grain is the only over-decomposition. | uneven sharding by relative units [pytorch#169320, `local_units`]; over-decomposition [Charm++ 1993] | rounding each extent separately, which breaks views and `from_local`'s inverse (global = local·P/p_r), so ranks infer different global shapes |
| R7 | The collectives a share waits for are priced inside the derivation, at the largest share. | communication time depends on the largest shard, and compute-proportional ratios lose where it dominates [HAP 2024, §2.4]; the ring's irregular all-gather is dominated by the largest block [Träff et al. 2010] | compute-proportional shares on communication-bound work |
| R8 | Equal counts run the regular collective. Unequal counts run the v-collective with each block at its own size, never padded to the largest. | Gatherv, Scatterv, Allgatherv, Alltoallv, Reduce_scatter [MPI 4.1 §6.5-6.10]; regular ⪯ irregular [Hunold & Carpen-Amarie 2017, GL4, GL8, GL12, GL18, GL22]; pad or per-rank broadcast [HAP 2024, §2.5.1; pytorch#198344] | padding's wasted bytes and window memory; the v-form's overhead on even splits |
| R9 | What a node can hold is a hard upper bound on its share. A share that exceeds it is clamped and the rest is solved again. | memory-bounded load balance [Whale 2022; Metis 2024]; GPU memory limits the batch [LB-BSP 2020]; bounded variables held at their bound and the rest re-solved [Bitran & Hax 1981] | a share the fast node cannot hold, which pages and makes the fast node slow |
| R10 | Nonlinear work is balanced on its own cost function (`min_max` takes any nondecreasing cost). Causal attention's work depends on where a share lies in the sequence; head-tail pairing makes it depend on the share alone. | causal ring attention's imbalance [Striped Attention 2023]; [Lastovetsky & Reddy 2007] | linear weights on quadratic work |
| R11 | Parts change with the configuration, not per step. An online loop has a dead band, a step below 1 and memory, and it acts only when the gain exceeds the cost of re-sharding. | static distributions avoid redistribution and control overhead [Beaumont et al. 2001, §2.2]; stop at a relative accuracy ε [DFPA 2011]; avoid over-reaction and oscillation [LB-BSP 2020, §3]; rebalance when it pays [Meta-Balancer 2012]; `allocate.Allocator` (forgetting factor, step η ≤ 1) | ping-pong after one noisy step; re-sharding weights because of jitter |
| R12 | Balance comes before overlap. Overlap hides a collective behind independent work, but it cannot hide a wait for a slower rank. Decomposed pieces follow the counts, and k pieces cost k·α. | overlap by decomposition [Wang et al. 2023; Domino 2024; FLUX 2024] | using overlap to cure an imbalance |
| R13 | Anything the partition cannot express falls back to stock's own fallback, or raises the same error on every rank. Nothing silently computes a wrong shape or value. | `AGENTS.md`: "A branch that quietly does less is worse than one that fails loudly" | silent numerics, such as a mean of unequal shards averaged as if the shards were equal |

## 4. The operand in use

The module docstring (`torch_mesh/partition.py`) is the API. This section is only its shape.

**Import.**
- On backend "mesh", no import is needed: `init_process_group(backend="mesh")`, which comes before
  any `DeviceMesh`, makes the backend's first process group, and that installs
  `torch_mesh.partition`.
- A program that imports `context_parallel_unshard` by name imports `torch_mesh.partition` first.
- On any other backend, import `torch_mesh.partition` before making the mesh.

**Operand.**
- `MESH_PARTITION="tp=3,5"` gives parts by mesh-dimension name and must be the same on every rank. It
  is attached when a `DeviceMesh` with that dimension name is made, and a sub-mesh keeps its root's
  parts.
- `partition.attach(mesh, tp=(3, 5))` does the same, before the mesh is first used.
- `partition.sizes(mesh, "tp", N)` gives the shares of N.
- A program takes its own splits from
  `distribute_tensor(x, mesh, [Shard(d)], src_data_rank=None).to_local()` or from `sizes()`, never
  from `x.chunk(world)`.
- `from_local` without `shape=` means "split by the parts".

**Derivation from the link map.** It runs once per configuration, by one procedure, on the driver
or identically on every rank:
1. For each mesh dimension, fix its grain (P), the operator class it runs, and that class's bytes and
   operations per unit at the call's shape.
2. `cost[r](c)` is the rate of rank r's node for that class times c units of that work. Find a
   rank's node by its host, never by assuming rank i is node i.
3. `shared(c)` is the time of the collectives on the dimension's critical path at the largest
   share: α + β·bytes for each, or `mesh_collective_time` with `segments` (`rdma/mesh.py`) where the
   map is more than a pair.
4. Call `allocate.min_max(P, 1, [1] * n, high, cost, shared)`, where `high[r]` is what the node can
   hold: its window over its bytes per unit.
5. Send the parts to every rank as `MESH_PARTITION`.

The rates go into the link map once, from a recorded run that the map cites. They change when the
configuration changes (R11), not with each run's timings.

**What stays uniform.**
- No operand, or equal parts: stock PyTorch.
- An extent that P does not divide: stock's split and stock's collectives.
- The strategy costs (`MeshTopoInfo`, `redistribute_cost`). They choose among collectives, not
  splits.
- What cannot take uneven shards falls back or raises; the docstring lists these cases.
- One operand per mesh dimension. There is no per-tensor override.
- Eager mode only.

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
- HexGen: Jiang, Y. et al. (2024). HexGen: generative inference of large language model over heterogeneous environment. ICML. https://arxiv.org/abs/2311.11514
- Hunold, S. & Carpen-Amarie, A. (2017). Tuning MPI collectives by verifying performance guidelines. https://arxiv.org/abs/1707.09965
- Ibaraki, T. & Katoh, N. (1988). Resource Allocation Problems: Algorithmic Approaches. MIT Press. https://dl.acm.org/doi/book/10.5555/49354
- Lastovetsky, A. & Reddy, R. (2007). Data partitioning with a functional performance model of heterogeneous processors. IJHPCA 21(1). https://doi.org/10.1177/1094342006074864
- LB-BSP: Chen, C., Weng, Q., Wang, W., Li, B. & Li, B. (2020). Semi-dynamic load balancing: efficient distributed learning in non-dedicated environments. SoCC. https://doi.org/10.1145/3419111.3421299
- MagPIe: Kielmann, T. et al. (1999). MagPIe: MPI's collective communication operations for clustered wide area systems. PPoPP. https://doi.org/10.1145/301104.301116
- Meta-Balancer: Menon, H., Jain, N., Zheng, G. & Kalé, L. (2012). Automated load balancing invocation based on application characteristics. IEEE Cluster. https://doi.org/10.1109/CLUSTER.2012.61
- Metis: Um, T. et al. (2024). Metis: fast automatic distributed training on heterogeneous GPUs. USENIX ATC. https://www.usenix.org/conference/atc24/presentation/um
- MPI Forum (2023). MPI: A Message-Passing Interface Standard, version 4.1, chapter 6. https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf
- Patarasuk, P. & Yuan, X. (2009). Bandwidth optimal all-reduce algorithms for clusters of workstations. JPDC 69(2). https://doi.org/10.1016/j.jpdc.2008.09.002
- pytorch#169320: [DTensor] A new placement: RaggedShard. https://github.com/pytorch/pytorch/issues/169320
- pytorch#198344: [c10d][NCCL] Uneven all_gather_into_tensor / reduce_scatter_tensor. https://github.com/pytorch/pytorch/issues/198344
- Roofline: Williams, S., Waterman, A. & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. CACM 52(4). https://doi.org/10.1145/1498765.1498785
- Song, L. et al. (2020). AccPar: tensor partitioning for heterogeneous deep learning accelerators. HPCA. https://doi.org/10.1109/HPCA47549.2020.00036
- Striped Attention: Brandon, W. et al. (2023). Striped attention: faster ring attention for causal transformers. https://arxiv.org/abs/2311.09431
- torch.chunk. https://docs.pytorch.org/docs/stable/generated/torch.chunk.html
- Träff, J. L., Gropp, W. & Thakur, R. (2010). Self-consistent MPI performance guidelines. IEEE TPDS 21(5). https://doi.org/10.1109/TPDS.2009.120
- Träff, J. L., Ripke, A., Siebert, C., Balaji, P., Thakur, R. & Gropp, W. (2010). A pipelined algorithm for large, irregular all-gather problems. IJHPCA 24(1). https://doi.org/10.1177/1094342009359013
- Whale: Jia, X. et al. (2022). Whale: efficient giant model training over heterogeneous GPUs. USENIX ATC. https://www.usenix.org/conference/atc22/presentation/jia-xianyan
- Wang, S. et al. (2023). Overlap communication with dependent computation via decomposition in large deep learning models. ASPLOS. https://doi.org/10.1145/3567955.3567959
