# Memorandum: sources, algorithms and data structures for ragged computation and unequal allocation

2026-11-09. Written by Claude at the operator's request, from the session
`.claude-personal/projects/-Users-mdot-metal-microbench/55974018-f56d-4c8b-8973-6fc61b49140c.jsonl`. It accompanies
[already-solved-subproblems-2026-11-09.md](already-solved-subproblems-2026-11-09.md), Parts 4 and 5.

Repository locations and uses of "HEAD" below belong to that dated inventory. They do not report the current tree
or establish behavior through an end-to-end run.

This is a list from which change requests could be written. It is not a change request: a change request's body is
the operator's turns, and each of its requirements quotes the source's own words
([change-requests.md](change-requests.md)).

## The turn

The operator, 2026-11-09:

> this oversteps; we are remarking that a specific list of functions are not needed for a specific list of feature work and change requests. instead, synthesize a positive list of definite citations and canonical algorihtms, methods, data structures, etc. which woudl form change-request-documents out of the 'requirements documents' which led to incomplete and self-inconsistent work wrt ragged computation, ragged tensors, etc.

## How definite each citation is

Each entry below carries one of these marks.

- **[H]** Cited by `design/heterogeneity.md`, with the reference in its References section. That document states
  what the source supports; Claude did not open the source in this session. A change request needs the source's own
  words, so each [H] is a pointer to read, not a quotation.
- **[Q]** Quoted verbatim from the source in another document of this directory, named with the entry.
- **[R]** Read against the source in this session, by the agent that wrote `~/schedule-emit`; the sections it read
  are named.
- **[S]** Reported by a search summary of the publisher's or project's page. The words in quotation marks are as
  the summary gave them; the source itself was not checked by that step.
- **[M]** From Claude's memory. Not a citation until someone reads it.

## 1. Data structures

### 1.1 The shares of a dimension, one integer per rank

- **What it is:** a mesh dimension's integers p_r ≥ 1, one per coordinate, of P = Σp units; rank r's share of an
  extent N is N·p_r/P (`design/heterogeneity.md` §1, "Parts", "Share", "Grain").
- **Sources:**
  - veScale's `RaggedShard(local_units)`: pytorch#169320, "[DTensor] A new placement: RaggedShard",
    https://github.com/pytorch/pytorch/issues/169320 **[H]**
  - Beaumont, Boudet, Petitet, Rastello and Robert, "A proposal for a heterogeneous cluster ScaLAPACK (dense linear
    solvers)", IEEE Trans. Computers 50(10), 2001, https://doi.org/10.1109/12.956091 **[H]**
  - Song et al., "AccPar: tensor partitioning for heterogeneous deep learning accelerators", HPCA 2020 (partition
    ratios) **[H]**; Zhang et al., "HAP", EuroSys 2024, §2.4 (sharding ratios per tensor and per layer) **[H]**
- **In these repositories:** `torch_mesh/partition.py` `RaggedShard`, `attach`, `split`, `from_local` (mesh be5d8920,
  history only); placement files with unequal integers (metal-microbench `tools/mesh/placements/`, at HEAD).

### 1.2 Counts and displacements of a collective, one per rank

- **What it is:** for a collective whose blocks differ in size, an array of each rank's element count and an array
  of where each rank's block begins.
- **Sources:**
  - MPI Forum, *MPI: A Message-Passing Interface Standard*, version 4.1, chapter 6: `MPI_GATHERV`, `MPI_SCATTERV`,
    `MPI_ALLGATHERV`, `MPI_ALLTOALLV`, `MPI_REDUCE_SCATTER`. `design/heterogeneity.md` cites §6.5–6.10 **[H]**. For
    `MPI_ALLGATHERV` the standard's page gives `displs` as "the displacement (relative to recvbuf) at which to place
    the incoming data from MPI process i", and says its outcome is as if every process called `MPI_Gatherv` for each
    root, https://mpi-forum.org/docs/mpi-4.1/mpi41-report/node126.htm **[S]**
  - pytorch#198344, "[c10d][NCCL] Uneven all_gather_into_tensor / reduce_scatter_tensor",
    https://github.com/pytorch/pytorch/issues/198344 **[H]**
- **In these repositories:** not found at any commit examined.

### 1.3 A tensor whose rows differ in length

- **What it is:** one flat array of values and one array of row boundaries.
- **Sources:**
  - TensorFlow, `tf.RaggedTensor`: `row_splits` is "a 1-D integer tensor with shape [nrows+1]", and "the values for
    row rt[i] are stored in the slice rt.values[rt.row_splits[i]:rt.row_splits[i+1]]"; several ragged dimensions
    are `nested_row_splits`, "ordered from outermost to innermost",
    https://www.tensorflow.org/api_docs/python/tf/RaggedTensor **[S]**
  - Apache Arrow, Columnar Format, "Variable-size List Layout": a validity bitmap, an offsets buffer and a child
    array, the offsets being indexes into the child array, https://arrow.apache.org/docs/format/Columnar.html
    **[S]**
- **In these repositories:** "Input domains are ragged relations: each rotated coordinate sums its own sources, no
  -1 padding" (mesh 444d4fd9, `torch_mesh/streamed.py` and `structured.py`; metal-microbench 3f308655; history
  only). A reader found there a relation of the form sources[starts[i]:starts[i+1]], which is this structure.

### 1.4 Parallel work over such a tensor, as flat operations

- **Sources:** Blelloch and Sabot, "Compiling collection-oriented languages onto massively parallel computers",
  Journal of Parallel and Distributed Computing 8(2), 1990: it "presents techniques to transform nested parallelism
  into flat, one-level parallelism" **[S]** (existence and subject only; nothing of its method was confirmed).
  Blelloch, *Vector Models for Data-Parallel Computing*, MIT Press, 1990 **[M]**.

### 1.5 A node's rates and a link's cost

- **What it is:** per node and per class of operator, a time model as a function of the call's shape; per direction
  of each pair of nodes, α per message plus β per byte.
- **Sources:**
  - Lastovetsky and Reddy, "Data partitioning with a functional performance model of heterogeneous processors",
    IJHPCA 21(1), 2007, https://doi.org/10.1177/1094342006074864 **[H]**
  - Williams, Waterman and Patterson, "Roofline", CACM 52(4), 2009 **[H]**; Jiang et al., "HexGen", ICML 2024,
    App. B **[H]**
  - Hockney, "The communication challenge for MPP", Parallel Computing 20(3), 1994 (the α-β model) **[H]**;
    Thakur, Rabenseifner and Gropp, 2005 (the α + nβ cost of collectives), quoted in
    `scaling-models-sources-2026-11-07.md` §3 **[Q]**
- **In these repositories:** metal-microbench `configs/links/pair-ring.txt` (parent of 8375cb22, history only).

### 1.6 What each processor's supplier offers

- **Sources:** Apple, Metal Feature Set Tables, as cited entry by entry in mesh `etc/compute/apple7.json` to
  `apple10.json` (mesh 90677543, history only). Claude read the first entries of one file only.

## 2. Algorithms

Each names the requirement of `design/heterogeneity.md` §3 it answers.

| # | what is computed | sources | requirement | in these repositories |
|---|---|---|---|---|
| 2.1 | integer shares that minimise the largest finish time: each grain goes to the rank that finishes earliest with it | Beaumont et al. 2001, Alg. 3.1 **[H]**; Ibaraki and Katoh, *Resource Allocation Problems: Algorithmic Approaches*, MIT Press, 1988 (greedy allocation is optimal for nondecreasing costs) **[H]** | R5 | `rdma/allocate.py` `min_max`, `equal_finish` (HEAD; imported by no tracked file) |
| 2.2 | the same under an upper bound per rank: a share over its bound is held there and the rest solved again | Bitran and Hax, Management Science 27(4), 1981 **[H]**; Whale 2022, Metis 2024, LB-BSP 2020 **[H]** | R9 | `rdma/allocate.py` `capacity`, and the bounds of `min_max` |
| 2.3 | shares for work that is not linear in the share | Lastovetsky and Reddy 2007 **[H]**; Brandon et al., "Striped attention", 2023 **[H]** | R10 | `min_max` takes any nondecreasing cost |
| 2.4 | the time of the collectives a share waits for, taken at the largest share | HAP 2024, §2.4 **[H]**; Träff, Ripke, Siebert, Balaji, Thakur and Gropp, "A pipelined algorithm for large, irregular all-gather problems", IJHPCA 24(1), 2010 **[H]**; Thakur, Rabenseifner and Gropp 2005 **[Q]** | R7 | not found (`min_max`'s `shared` argument has no caller) |
| 2.5 | a collective with each block at its own size | MPI 4.1 ch. 6 **[H]**, **[S]**; Träff et al. 2010 **[H]**; the alternative of padding, HAP 2024 §2.5.1 **[H]** | R8 | not found |
| 2.6 | the same counts on every rank, from one procedure over shared configuration | MPI 4.1 §6.7, §6.10.2 **[H]**; Lastovetsky and Reddy, "HeteroMPI", JPDC 66(2), 2006 **[H]** | R4 | placement files are shared data |
| 2.7 | tensors that meet in one operation split their shared extent alike | Shoeybi et al., "Megatron-LM", 2019, §3 **[H]** | R6 | `prepare_e2b.py` `column_cuts`, `shard_heads` (HEAD) |
| 2.8 | fixed shapes under changing shares: pad to the largest share and mask the padding to the identity of the next operation | Xu et al., "GSPMD", 2021, §3.3 **[H]**; GSPMD is also quoted in `joint-optimization-sources-2026-11-07.md` §5 **[Q]** | R14 | `_partition_pad` emits the pad; a reader found no masking |
| 2.9 | which collective algorithm a link map calls for | Kielmann et al., "MagPIe", PPoPP 1999; Cho et al., "BlueConnect", 2019; Wang et al., "Blink", 2020 **[H]**; Patarasuk and Yuan, JPDC 69(2), 2009 (in a ring every block crosses n−1 of n links) **[H]** | R2 and §2 | `mesh_collective_choose` not found at HEAD |
| 2.10 | when shares change | Beaumont et al. 2001, §2.2 **[H]**; LB-BSP 2020, §3.2.1 and §3.3.2 **[H]**; DFPA 2011 **[H]**; Menon et al., "Meta-Balancer", 2012 **[H]** | R11 | `allocate.Allocator` (a forgetting factor and a step) |
| 2.11 | overlap of a collective with independent work, after balance | Wang et al., ASPLOS 2023; Domino 2024; FLUX 2024 **[H]** | R12 | — |
| 2.12 | the order of functions on processors of different speeds | Topcuoglu, Hariri and Wu, "Performance-effective and low-complexity task scheduling for heterogeneous computing", IEEE TPDS 13(3), 2002: §2 (a cost of each function on each processor; earliest start and finish times), §4.1 (upward rank, eq. 8–9, on mean costs), §4.2 (order by nonincreasing rank; each function to the processor of earliest finish; insertion into the first idle slot) **[R]** | scheduling | `~/schedule-emit/schedule_emit/schedule.py` takes the rank and the insertion and not the choice of processor |
| 2.13 | functions called different numbers of times in one period | Lee and Messerschmitt, "Static Scheduling of Synchronous Data Flow Programs for Digital Signal Processing", IEEE Trans. Computers C-36, 1987: the topology matrix, the repetitions vector, a periodic admissible schedule **[S]** | scheduling | — |
| 2.14 | storage and waits between processors under a compiled order | Bhattacharyya, Sriram and Lee, IEEE TSP 45(6), 1997: §3, §4.1, §4.3 eq. 5, §5.1 **[R]**; Lam, PLDI 1988, §2.3 **[R]** | scheduling | `mesh_graph_buffer_bound`; `~/schedule-emit` `argument_buffers` |
| 2.15 | shares and the schedule chosen together | Alpa, Unity, CoCoNet, TACCL, quoted in `joint-optimization-sources-2026-11-07.md` §1–4 **[Q]**; HAP 2024 (program synthesis) **[H]** | R7 | `design/joint-compilation.md` |
| 2.16 | many more pieces than processors, placed while running (the alternative `design/heterogeneity.md` names and does not build) | Kalé and Krishnan, "CHARM++", OOPSLA 1993 **[H]** | — | not built |

## 3. Where the requirement documents are incomplete or disagree with themselves or with the code

These are the dated inventory's source comparisons and reports of what its readers did not find.

1. **Named files that do not exist at HEAD.** `design/heterogeneity.md` names metal-microbench
   `configs/links/pair-ring.txt` (removed by 8375cb22, 11-03), `tools/torch_parallel/parts.py` and `pair.py` (no
   tracked source), and `torch_mesh/partition.py` as "the API" (78 lines at mesh 90677543; untracked at HEAD).
2. **Named functions that were not found.** `mesh_collective_time`, `mesh_collective_choose`, `partition.write`,
   `partition.sizes`.
3. **A signature that differs from the code.** §4 writes `partition.attach(mesh, tp=(capacity, parts))`; the last
   code read has `attach(mesh, tp=units)`, an integer, and derives the parts itself.
4. **R8 and the compiler.** R8: "Unequal counts run the v-collective with each block at its own size, never padded
   to the largest." `rdma/mesh_compile.py` `_reshard` and `_partition` pad a dimension to a multiple of the mesh
   extent before a collective.
5. **R8 beside R14.** R8 forbids padding a collective's blocks; R14 shapes each rank's local buffer "to the largest
   share the bounds allow it" and keeps padding masked. The two concern different things, the bytes sent and the
   buffer held, and the document does not say so in either row.
6. **R14 and the code.** R14 requires padding to be re-masked to the identity of its next consumer. A reader found
   the pad and no masking in the functions it read.
7. **R6 and R7 beside the module's stated limit.** The sources' ratios are per tensor and per layer; "One set of
   parts per mesh dimension ... The design's optional per-tensor override is not built."
8. **R11.** It lists what an online loop needs and then says `allocate.Allocator` "has a forgetting factor and a
   step ... and none of the rest."
9. **R7's price.** Step 3 of the derivation defines `shared(c)`; no function computes it, and `min_max`'s `shared`
   argument has no caller.
10. **Ragged tensors are not specified.** `design/heterogeneity.md` specifies unequal shards across ranks. Rows of
    differing length within one tensor (1.3) appear in commit messages (mesh 444d4fd9, be5d8920, 967ed4ac) and in
    no requirement document that was found.

The resulting assignment is [unequal-shards-change-request.md](unequal-shards-change-request.md), including
its requirements for ragged values, placement, masking and collectives.
