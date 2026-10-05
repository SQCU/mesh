# Heterogeneous nodes and links

Orientation for anyone, developer or agent, who splits work over nodes of different speeds or
moves it over links of different costs: `rdma/torch-mesh/torch_mesh/partition.py` (the partition
operand), `rdma/allocate.py` (the shares), `rdma/mesh-plan.h` and `mesh-collective.c` (the compiled
collectives), metal-microbench `tools/mesh/calibrate.py` (the rates and link costs) and the programs that
use them. The operator:

> "it'll take a little bit of orientation for developers (agents as well) to handle heterogenous
> links ontologically (since this is so different) and without lots of clumsy erratic
> overcorrections (this is mostly requirements in the literature, yo)."

None of it is new. Stock PyTorch splits a sharded dimension equally (`Shard` follows `torch.chunk`),
a uniform prior over the ranks. Splits in proportion to device speed are heterogeneous ScaLAPACK's
[Beaumont et al. 2001], AccPar's partition ratios [Song et al. 2020], HAP's sharding ratios
[Zhang et al. 2024] and veScale's `RaggedShard(local_units)` [pytorch#169320]. The collectives they
need are MPI's v-collectives [MPI 4.1 §6.5-6.10].

## 1. Vocabulary

- **Rate**: a node's time model for one operator class at one dtype, a function of the call's
  shape. It is written on the node's line of the link map as that class's constants, in the form
  the map's node-line comment gives (metal-microbench `configs/links/pair-ring.txt`: `proj=u,v`, u ns
  per weight byte plus v ns per weight element per row, for the engine's fp16 projections;
  `attn=u,v`, u + v·bs ns per byte of one cached position read at batch bs). `calibrate.py node`
  writes a node's line from its probes. A class the map does not carry (a program's fp32 MPS matmul,
  its attention) is measured on each node and added as its own class (R3). A node has one rate per
  class and dtype, never one speed.
- **Link**: the cost of one direction of one pair of nodes, α (µs) per message plus β (ns) per byte.
  It is written on a link line of the link map.
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
| Node rates | the link map's node lines | the derivation (`allocate.equal_finish`; this repository's models: metal-microbench `programs.py derive`) | the parts: what each node computes |
| Links | the link map's link lines, each with its own cost | the compiled collectives (`mesh_compile` along shortest-path or packed trees, the trees packed by each link's cost; libnccl-mesh's default table or a program's own, `ncclMeshCommInitRank`, torch-mesh `Options`); the derivation, only to price collectives (R7) | the single-phase all-reduce or reduce-scatter and all-gather, and their trees: how operands travel |

- Rates decide the parts. The link map decides the algorithm.
- A link's cost never becomes a node's rate, and a node's rate never selects an algorithm.
- The one coupling is R7: the derivation prices the collectives a share waits for, and they cost
  most at the largest share, so where communication dominates the shares move toward even
  [HAP 2024, §2.4].
- Shifting shares cannot route around a slow link. In a ring, every block crosses n−1 of the n
  links [Patarasuk & Yuan 2009], so a slow link carries every block but one whatever the shares.
  The shares change only how many bytes it carries (N minus the destination's share), and R7
  prices those.

## 3. Requirements

| | Requirement | Source | Overcorrection it prevents |
|---|---|---|---|
| R1 | No operand means stock PyTorch, bit for bit. Equal parts also mean stock, including `torch.chunk`'s ceil sizes (10 over 4 is 3,3,3,1, not largest remainder's 3,3,2,2). | [torch.chunk]; the regular operation is the irregular one's special case and no slower than it [Träff, Gropp & Thakur 2010; Hunold & Carpen-Amarie 2017, GL4] | a second "uniform" rule that silently changes programs that never set an operand |
| R2 | What each node computes follows from the nodes' rates. How operands travel follows from the link map. The only coupling is R7's price of the collectives. | metal-microbench `docs/principles.md` §4; link-aware collectives [MagPIe 1999; BlueConnect 2019; Blink 2020] | a node's rate measured with its collectives inside it, so a slow link reads as a slow node; reshaping collectives to "fix" a slow node |
| R3 | Rates are per node **and per operator class** at the program's dtype, taken at the call's shape (bytes and operations), from a functional model. | functional performance model [Lastovetsky & Reddy 2007]; [Roofline 2009]; per-device memory bandwidth and FLOPS in the cost model [HexGen 2024, App. B] | one speed per node applied to every operator; bandwidth-bound (decode) rates applied to compute-bound work; another class's or dtype's rate borrowed where the map has none |
| R4 | Counts are derived once per configuration, from shared configuration, by one deterministic procedure, so they are identical on every rank. | counts agree across ranks: recvcounts are the same on all members [MPI 4.1 §6.7, §6.10.2]; the distribution follows from a declared performance model [HeteroMPI 2006] | counts from each rank's own timing: ranks that disagree hang or corrupt |
| R5 | Integer rounding minimises the largest finish time: each grain goes to the rank that finishes earliest with it. | [Beaumont et al. 2001, Alg. 3.1]; greedy allocation is optimal for nondecreasing costs [Ibaraki & Katoh 1988] | largest remainder rounding the slow rank up by a whole grain |
| R6 | Tensors that meet in one operation split their shared extent alike, and every rank can invert its own split from its local extent and the capacities (`from_local`: global = local·P/c_r, R14). | the first GEMM's column split is the second's row split [Megatron-LM 2019, §3]; the literature's ratios are per tensor, `RaggedShard(local_units)` on each placement [pytorch#169320], and per layer [HAP 2024, §2.4] | rounding each extent separately, or splitting an extent the parts do not divide by `torch.chunk`: views and the inverse break, and ranks infer different global shapes |
| R7 | The collectives a share waits for are priced inside the derivation, at the largest share. | communication time depends on the largest shard; even ratios win "when communication is the bottleneck", and since layers differ in computation-to-communication ratio, the optimal ratios may vary for each layer [HAP 2024, §2.4]; the ring's irregular all-gather is dominated by the largest block [Träff et al. 2010] | compute-proportional shares on communication-bound work |
| R8 | Equal counts run the regular collective. Unequal counts run the v-collective with each block at its own size, never padded to the largest. | Gatherv, Scatterv, Allgatherv, Alltoallv, Reduce_scatter [MPI 4.1 §6.5-6.10]; regular ⪯ irregular [Hunold & Carpen-Amarie 2017, GL4, GL8, GL12, GL18, GL22]; pad or per-rank broadcast [HAP 2024, §2.5.1; pytorch#198344] | padding's wasted bytes and window memory; the v-form's overhead on even splits |
| R9 | What a node can hold is a hard upper bound on its share. A share that exceeds it is clamped and the rest is solved again. | memory-bounded load balance [Whale 2022; Metis 2024]; GPU memory limits the batch [LB-BSP 2020]; bounded variables held at their bound and the rest re-solved [Bitran & Hax 1981] | a share the fast node cannot hold, which pages and makes the fast node slow |
| R10 | Nonlinear work is balanced on its own cost function (`equal_finish` takes affine costs a + b·s; `min_max` any nondecreasing cost, and `dfpa` balances on measured costs, built 2026-10-05). Causal attention's work depends on where a share lies in the sequence; head-tail pairing makes it depend on the share alone. | causal ring attention's imbalance [Striped Attention 2023]; [Lastovetsky & Reddy 2007] | linear weights on quadratic work |
| R11 | Parts change with the configuration, not per step. An online loop follows its sources: it predicts robustly and reduces its step once it oscillates, stops at a relative accuracy, and rebalances only when the gain exceeds the cost. `allocate.dfpa` has all of it: a three-call median window, half steps after a reversal, a stop at a relative accuracy, and a rebalance only where the measured models promise more than it (metal-microbench `tools/mesh/rates.py` runs it after every ledgered call; `tools/mesh/balance_sim.py` validates it on simulated M1-M12 and counterfactual nodes). `allocate.Allocator` (a forgetting factor and a step η) remains for its own callers. | static distributions avoid redistribution and control overhead [Beaumont et al. 2001, §2.2]; prediction robust to non-deterministic perturbation "to avoid over-reaction or oscillation" [LB-BSP 2020, §3.2.1], an observation window and a smaller step once oscillation is detected [LB-BSP 2020, §3.3.2]; stop at a relative accuracy ε [DFPA 2011]; invoke the balancer when the gain exceeds its cost [Meta-Balancer 2012] | ping-pong after one noisy step; re-sharding weights because of jitter |
| R12 | Balance comes before overlap. Overlap hides a collective behind independent work, but it shortens a rank's own finish, not its wait for a slower rank. Decomposed pieces follow the counts, and k pieces cost k·α. | overlap by decomposition [Wang et al. 2023; Domino 2024; FLUX 2024]; balance first because the step ends at the largest finish time (§1); k·α in the α-β model [Hockney 1994] | using overlap to cure an imbalance |
| R13 | Anything the partition cannot express falls back to stock's own fallback, or raises the same error on every rank. Nothing silently computes a wrong shape or value. | `AGENTS.md`: "A branch that quietly does less is worse than one that fails loudly" | silent numerics, such as a mean of unequal shards averaged as if the shards were equal |
| R14 | Parts change without shapes changing. Each rank's local buffer is shaped to the largest share the bounds allow it (its capacity, stated with the parts); the parts are a tensor operand read when the ops run. Padding holds the identity of its next consumer: sources write it (weights zero-padded at load, collectives the identity past the valid extent), and before any op that reduces or contracts over a partitioned dimension the shard is re-masked to that op's identity (0 for sums and contractions, -inf for max and masked softmax); attention masks keys past the valid extents; count-dependent ops divide masked sums by the valid count. Kernels are stock: they compute garbage in padding that nothing reads unmasked. | uneven partitioning: pad to the shards' size and mask the padding to the identity of the next operation wherever it could leak into valid results [GSPMD 2021, §3.3] | shapes that follow the parts, so every change of parts recompiles, re-records and reallocates a program |

## 4. The operand in use

The module docstring (`torch_mesh/partition.py`) is the API. This section is only its shape.

**Import.**
- On backend "mesh", no import is needed: `init_process_group(backend="mesh")`, which comes before
  any `DeviceMesh`, makes the backend's first process group, and that installs
  `torch_mesh.partition`.
- A program that imports `context_parallel_unshard` by name imports `torch_mesh.partition` first.
- On any other backend, import `torch_mesh.partition` before making the mesh.

**Operand.**
- `partition.attach(mesh, tp=(capacity, parts))`, before the mesh is first used, the same on every
  rank; a sub-mesh keeps its root's. `partition.write(mesh, tp=parts)` changes the parts between calls.
- `partition.sizes(mesh, "tp", N)` gives the local buffers' sizes of N (R14).
- A program takes its own splits from
  `distribute_tensor(x, mesh, [Shard(d)], src_data_rank=None).to_local()` or from `sizes()`, never
  from `x.chunk(world)`.
- `from_local` without `shape=` means "a capacity block" (global = local·P/c_r).
- Every extent split along a partitioned dimension is a multiple of P; any other raises
  `ValueError`, on every rank alike. No mesh dimension of more than one coordinate precedes a
  partitioned one (attach raises otherwise), so no outer split reaches a partitioned dimension first
  and the extent it splits is the global one: DTensor splits a tensor dimension sharded on several
  mesh dimensions outer first, and an outer split's shares can differ by coordinate, so ranks would
  disagree about P dividing them. To partition an inner dimension, order the mesh with it first.

**Derivation from the link map.** It runs once per configuration, by one procedure, on the driver
or identically on every rank:
1. For each mesh dimension, fix its grain (P), the operator classes it runs (a TP dimension runs the
   QKV and output projections, the FFN and attention), and each class's work per unit at the call's
   shape. These are stated for the program, never read from the ops it dispatches.
2. Each rank's time for c units is a_r + b_r·c: b_r the sum over those classes of the class's rate on
   rank r's node times its work per unit, a_r the fixed terms counted once. The group's nodes in rank
   order are the process group's `node` (torch-mesh reads each rank's from its bridge); never infer a
   node from a host name or assume rank r is node r.
   - If the map has no rate for a class at the program's dtype and shape, measure it on each node
     and add it to the node lines as its own class, with the record it came from, before deriving.
     Never borrow another class's rate. The first weighted run on the pair derived fp32 MPS
     programs' parts from the engine's fp16 projection rates (4.1x per FMA between the nodes, where
     these programs' matmuls run 2.6x), and the wait moved to the slower node (metal-microbench
     `docs/measurement.md` ledger, 2026-09-28).
3. Add to a_r the time of the collectives on the dimension's critical path at the largest share: the
   compiled program's time on the map, `mesh.program_time(map, program, operand)` (R7).
4. `allocate.equal_finish(P, 1, low, high, a, b)` gives the integer parts, `high[r]` what the node can
   hold (its memory over its bytes a unit, R9).
5. Give every rank the parts and the capacities to attach (`partition.attach(mesh, tp=(capacity,
   parts))`); metal-microbench `tools/torch_parallel/tp.py` takes them as `PARTITION`.

The rates and the link costs go into the link map once, from recorded runs the map cites
(`calibrate.py node` and `calibrate.py links`). They give the first parts only.

**Parts from the run's evidence (R3, R10, R11).** `partition.rebalance(mesh, name, every)`, called once a step on
every rank, watches the dimension's group (`torch_mesh/evidence.py`: each rank's own work between its collectives,
timed on torch's MPS stream or the host clock, never a collective's wait, R2). Every `every` steps it gathers the
ranks' work to all of them and runs one step of `allocate.Balancer` (DFPA on the measured points), identically on
every rank (R4). Its parts stand once the evidence promises no gain past its resolution, so they change with the
configuration and not with jitter (R11). A rank's work follows its capacity (R14: stock kernels compute the
padding), so parts that move past the capacities are attached again at them; that is the one cost R11 weighs.
metal-microbench's engine calls run the same Balancer on their bridges' ledgers (`tools/mesh/rates.py`).

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
- HexGen: Jiang, Y. et al. (2024). HexGen: generative inference of large language model over heterogeneous environment. ICML. https://arxiv.org/abs/2311.11514
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
- Roofline: Williams, S., Waterman, A. & Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. CACM 52(4). https://doi.org/10.1145/1498765.1498785
- Song, L. et al. (2020). AccPar: tensor partitioning for heterogeneous deep learning accelerators. HPCA. https://doi.org/10.1109/HPCA47549.2020.00036
- Striped Attention: Brandon, W. et al. (2023). Striped attention: faster ring attention for causal transformers. https://arxiv.org/abs/2311.09431
- torch.chunk. https://docs.pytorch.org/docs/stable/generated/torch.chunk.html
- Träff, J. L., Gropp, W. & Thakur, R. (2010). Self-consistent MPI performance guidelines. IEEE TPDS 21(5). https://doi.org/10.1109/TPDS.2009.120
- Träff, J. L., Ripke, A., Siebert, C., Balaji, P., Thakur, R. & Gropp, W. (2010). A pipelined algorithm for large, irregular all-gather problems. IJHPCA 24(1). https://doi.org/10.1177/1094342009359013
- Whale: Jia, X. et al. (2022). Whale: efficient giant model training over heterogeneous GPUs. USENIX ATC. https://www.usenix.org/conference/atc22/presentation/jia-xianyan
- Wang, S. et al. (2023). Overlap communication with dependent computation via decomposition in large deep learning models. ASPLOS. https://doi.org/10.1145/3567955.3567959
