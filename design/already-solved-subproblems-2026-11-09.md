# Memorandum: candidate implementations found for the subproblems on 2026-11-09

2026-11-09. Written by Claude at the operator's request, from the session
`.claude-personal/projects/-Users-mdot-metal-microbench/55974018-f56d-4c8b-8973-6fc61b49140c.jsonl`. It accompanies
[implementation-order-2026-11-09.md](implementation-order-2026-11-09.md) and
[interface-boundaries-2026-11-09.md](interface-boundaries-2026-11-09.md).

Code was read at mesh 7df87fe4 and at the metal-microbench working tree on 7483572e.
Locations and uses of "HEAD" and "now" below belong to that dated inventory. The inventory did not establish
which implementations meet their requirements through the sanctioned end-to-end calling paths.

## The turns

The operator, 2026-11-09, in order:

> we, regardless, might still need to do a lot of function composition and loop analysis and functions: `f(IR)->IR` to turn arbitrary pytorch or whatever it is litert does to mps-spec-appropriate use of the basic features in mps. this doesn't mean we need to define numerical backends, but it does give us some really obvious responsibilities which kept getting conflated with reimplementing numerics (for some reason...?) in previous implementation passes... or maybe the user is totally wrong about this somehow? discuss.

> discuss the possibility of a function over tensors which is sharded to ANE on one node, CPU-SIMD on another node, and MPS on a third, fourth, fifth, and sixth node.

> how many of these subproblems have already been solved piecemeal and with incorrect data flow placement iwthin the last 2 weeks of commits? we need to draw up a manifest of 'hwo already-solved all of the subproblems this repo really needed solved' are. document the report from this previous turn...

## Part 1. The report of the previous turn

The original report mixed cited transformations with claims from memory about Apple's and LiteRT's interfaces;
no header was checked. The unsupported supplier-only and processor restrictions were removed on 2026-11-13.

### Functions from IR to IR

The report identified these transformations between an exported graph and numeric functions:

1. **Selecting, for each operator, what the supplier offers.** Instruction selection at the level of tensor
   operators: MLIR's dialect conversion (compilation-composition-change-request.md, S2).
2. **Composing operators into one function per launch.** Operator fusion: TVM section 3, quoted in
   torch-source-lowering-change-request.md item 1 ("Operator fusion combines multiple operators into a single kernel
   without saving the intermediate results in memory").
3. **Loop and index analysis.** Which part of an operator's index space a shard or a page covers, and which pages a
   function reads: tiling and bounds inference (S1 and S9), and exact dependence
   (compilation-composition-change-request.md, "Source bindings for compact effects and scalar lowering").
4. **View and layout analysis,** so an operand is bound by view and a copy appears only as a node: bufferization
   (S8).
5. **Folding static operands into the program.** Partial evaluation (S6).

The turn quoted by torch-source-lowering-change-request.md item 1 permits source from either place:
"prohibiting the use of any 'torch metal kernels' rather
than a lowering through source, any source, it can be reference soruce, it can be our source, i don't care".

### One function sharded across ANE, CPU and four MPS nodes

The operator's question above is the scope of this inventory. This memorandum did not measure such a program or
establish the capabilities, precision or execution path of those processors.

## Part 2. The manifest

How this was made: for each subproblem, the function names were found by searching the source, and the first commit
containing each name by `git log -S`. These are candidate locations: this search did not read the bodies, run them,
or establish their purpose, correctness or coverage. Where a function sits in the data flow is taken from
[interface-boundaries-2026-11-09.md](interface-boundaries-2026-11-09.md). In the fourteen days before this
memorandum the two repositories had 1,869 commits (mesh) and 642 (metal-microbench).

### The five functions from IR to IR

| subproblem | candidate locations | first commit | reported location at the dated revision | proposed stage |
|---|---|---|---|---|
| 1. selecting what the supplier offers | `rdma/mesh_compile.py`: `_contract`, `_reduced`, `_selected`, `_lowered`, `_decomposed`; `provider/litert_mesh.cc`: the lowering of LiteRT nodes; `metal_neural.mm`: `LowerNeural` | 11-07 (`_contract`), 11-09 (the rest) | in three modules, each also emitting the bodies of numeric functions | one function on tensor operators, before scheduling |
| 2. composing operators into one function per launch | `metal_transform.mm`: `Fuse`, `Inline`, `Epilogue`, `Groups`, `Profitable`; `rdma/mesh_compile.py`: `_epilogue` | 11-09 | on Tint instructions; `metal_transform.mm` calls `Map`, `Bindings` and `TransportBindings`, which the lowering and transport files define | on tensor operators, before scheduling; the supplier's compiler makes one launch of the composed function |
| 3. loop and index analysis | `rdma/mesh_compile.py`: `_extent`, `_paged`, `_partition`'s regions; `rdma/mesh-graph.h`: `mesh_graph_intersect`, `mesh_graph_dependences_views`, `mesh_graph_conflicts_views`; `metal_ir.mm`: `Footprints`, `Effects`, `Relation` | 11-05 (`_extent`), 11-06 (`mesh_graph_intersect`), 11-09 (the rest) | computed in the frontend, again in the graph module, and again in the compiler core | once, in the graph module, with the frontend supplying the accesses |
| 4. view and layout analysis | `rdma/mesh_compile.py`: `_view`, `_view_rule`, `_aliases`, `_layout`, `_strided`, `_contiguous`; `metal_ir.mm`: `Contiguous`; `provider/litert_mesh.cc`: `Bind`'s strides | 11-06 (`_aliases`), 11-09 (`_view_rule`) | in the frontend, the compiler core and the provider | once, on the graph, before scheduling |
| 5. folding static operands | `rdma/mesh_compile.py`: `_specialize`, `_fold`, `_constant_origins`, `_constant_storage`; `metal_transform.mm`: `Constants`, `Fold`, `Prune`, `Simplify`, `ZeroPoints`; `metal_lower.mm`: `PrepareConstants` | 11-08 (`_specialize`), 11-09 (the rest) | in the frontend and on Tint instructions; `PrepareConstants` compiles a function and calls it while lowering | once, on tensor operators, before scheduling |

### The stages of the implementation order

| subproblem | candidate locations | first commit | reported location at the dated revision |
|---|---|---|---|
| sharding a tensor across a collective, with sends and receives | `rdma/mesh_compile.py`: `partition`, `_partition`, `_reshard` (GSPMD is cited there) | 11-04 | the frontend |
| a placement operand, uneven splits included | `tools/mesh/placements/even-5.json`, `attn-7-1-mlp-ledger-bal.json` | 11-08 (`even-5.json`) | data, as required |
| one scheduling solve | `rdma/mesh-graph.h`: `mesh_graph_schedule`, `mesh_graph_list_schedule`, `mesh_graph_heap_order`, `mesh_graph_completion_times`; `rdma/mesh-collective.c`: `plan_schedule` | 10-30, 11-03, 11-04 | the plan's owner, for transfers (`mesh_graph_schedule` itself has no call site). Solved again for functions in `metal_ir.mm` `Order` (11-09), `metal_native.mm` `Choose` (11-09) and `metal_recording.m` `MetalProgramPrepare` |
| storage: lifetimes, copies of a value, buffer bounds | `rdma/mesh-graph.h`: `mesh_graph_buffer_bound`, `mesh_graph_lags`, `mesh_graph_unfold`, `mesh_graph_greedy_by_size`; `rdma/mesh_compile.py`: `_lifetimes`, `copies` | 11-08, 11-09 | the graph module and the frontend; derived again when a program is prepared or loaded |
| cost of a function on a processor | `rdma/mesh-graph.h`: `mesh_graph_actor_time`, `mesh_graph_cost_times`; `rdma/mesh_compile.py`: `_actors`; `metal_ir.mm`: `Actor`; `metal_transform.mm`: `Time`, `Profitable` | 11-09 | the graph module, the frontend, the compiler core and the fusion pass |
| saving and loading | `rdma/mesh_compile.py`: `_save_program`, `_load`; `metal_ir_save`, `metal_ir_load`; the save and load functions of `metal_lower.mm`, `metal_neural.mm`, `metal_recording.m` | 11-09 | about 19 functions in five files |
| the synchronization signature | `rdma/mesh-graph.h`: `mesh_graph_signature*`; `rdma/mesh_graph.py`: `signature`, `record` | before 11-09 | the graph module; assembled by the runtime |

### Not found at HEAD

- **Which operators each processor's supplier offers, as data.** No such table is in the tracked tree.
  `metal_native.mm` reads the machine's core count and GPU family from the system itself (`metal_native.mm:28-33`,
  `:374`), and metal-microbench `kernel_capabilities.json` lists quantization formats, not machines. It was written
  and is in history: see Part 5 (`etc/compute/`).
- **The macOS build and the Metal language version as operands.** `bin/mesh-nodeinfo.sh` reads the macOS version;
  it was not found in the node record or passed to the compiler.

### The count

The tables list candidates for five IR transformations and seven stage subproblems. Two operands were not found
by that search. These counts do not establish that the problems are solved, that the methods are complete, or
which bodies to keep. Part 3 records the later readers' findings and limits.

Corrected after Part 3's reading: an earlier form of this table listed `mesh_graph_compose` and
`mesh_graph_contract` under composing operators. They compose synchronization signatures and contract dependence
edges; neither composes the bodies of operators.

## Part 3. What reading the bodies showed

The operator, 2026-11-09, quoting the limit stated above:

> "- "Written" means a function of that purpose exists. I did not read the bodies, run them, or establish their correctness or coverage.' use haiku-parallel repoistory parsing yet again...

Eleven readers (Claude Haiku), one per subproblem, read the bodies of 209 functions. What each function does, takes,
returns, cites and refuses is in
[subproblem-function-bodies-2026-11-09.md](subproblem-function-bodies-2026-11-09.md), with what Claude checked and
which of the readers' statements are unreliable. Reading a body supplies source observations; it does not
establish reachability, executed behavior or correctness. Nothing was run.

| subproblem | functions read | whole body read | cite a source in the code | cases refused | same thing computed in more than one place | parts with no function |
|---|---:|---:|---:|---:|---:|---:|
| selecting what a supplier offers | 22 | 15 | 14 | 30 | 6 | 6 |
| composing operators into one function per launch | 11 | 11 | 10 | 43 | 6 | 7 |
| loop and index analysis | 19 | 16 | 13 | 19 | 7 | 5 |
| view, layout, stride and alias analysis | 15 | 14 | 8 | 14 | 6 | 3 |
| folding static operands | 21 | 19 | 9 | 13 | 5 | 5 |
| sharding, with sends and receives | 12 | 10 | 8 | 9 | 3 | 4 |
| the scheduling solve | 14 | 14 | 11 | 54 | 6 | 5 |
| storage decisions and cost | 18 | 17 | 16 | 30 | 5 | 6 |
| saving, loading and the signature | 37 | 30 | 14 | 81 | 8 | 5 |
| allocating unequal work | 19 | 17 | 13 | 24 | 4 | 13 |
| ragged tensors and relations | 21 | 15 | 14 | 17 | 5 | 7 |

What the reading changes in Part 2:

- **Selecting what a supplier offers is not written as selection.** `_lowering` (`rdma/mesh_compile.py:6694-6723`)
  takes the first matching branch of an ordered chain and raises for anything else ("its lowering into the
  structured basis is unfinished"). No function chooses among suppliers by cost, device or measurement.
- **Composition is written for one case.** `Inline` and `Epilogue` (`metal_transform.mm`) fuse one producer into
  one consumer, on the same processor, for operations without carried state; other cases are refused there.
- **Written but not called.** `mesh_graph_schedule` and `mesh_graph_unfold` have no call site. The ordering used on
  the Metal side is `mesh_graph_levels` and `mesh_graph_heap_order` from `metal_ir.mm`.
- **Sharding pads to equal shards.** `_reshard` and `_partition` pad a dimension to a multiple of the mesh extent
  (`rdma/mesh_compile.py:715`, `:745`, `:903`). No function read emits a collective with a size per rank.

The operator, 2026-11-09, on these findings:

> so we know we will never need to read or cite the internals of these unimplemented functions, throw that in the manifest i suppose

That remark is narrow. It concerns the specific functions this part names, and the specific work this manifest
serves:

- `_lowering`'s ordered chain (`rdma/mesh_compile.py:6694-6723`), for selecting among suppliers;
- `Inline` and `Epilogue` (`metal_transform.mm`), for composing operators beyond the one case they cover;
- `mesh_graph_schedule` and `mesh_graph_unfold`, which have no call site;
- the padding in `_reshard` and `_partition`, for unequal shards.

For the items of implementation-order-2026-11-09.md and for the change requests they quote, the internals of these
functions need not be read or cited. Nothing is claimed here about any other function, or about these functions for
any other purpose. An earlier form of this paragraph stated a general rule; the operator, 2026-11-09: "this
oversteps; we are remarking that a specific list of functions are not needed for a specific list of feature work
and change requests."

The positive counterpart, for ragged computation and unequal allocation, is
[ragged-computation-sources-2026-11-09.md](ragged-computation-sources-2026-11-09.md): the citations, algorithms and
data structures a change request would be written from.

## Part 4. Ragged tensors, and unequal work across nodes of different speeds

The operator, 2026-11-09:

> while youj're at it pull up all of the resources the repo already wrote / designed / solved for problems of ragged tensors, parametrically allocating compute-ragged and scheduling-ragged allocations of work across heterogenous nodes, etcetera, whihc we will ultimately need as part of this 'manifest of solved algorithmic problems whether or not their literal implementations cmopose or comport with our requirements'

Much of this was written in October and is no longer in the tracked tree: mesh 0b3e78d7 (11-03) "mesh returns to
3981ca80 (2026-10-01)" and metal-microbench 864a0ae4 (11-03) returned both repositories to their state of
2026-10-01. What follows says, for each resource, whether it is at HEAD or only in history.

### The specification

`design/heterogeneity.md` (mesh, at HEAD, 186 lines) is the statement of the problem and its sources. It opens with
the operator's words: "it'll take a little bit of orientation for developers (agents as well) to handle heterogenous
links ontologically (since this is so different) and without lots of clumsy erratic overcorrections (this is mostly
requirements in the literature, yo)."

- **Vocabulary** (section 1): rate (a node's time model for one class of operator at one type), link (α per message
  plus β per byte), parts, share, grain, counts, finish time.
- **Two facts, two mechanisms** (section 2): "Rates decide the parts. The link map decides the algorithm."
- **Fourteen requirements** (section 3), each with its source and the overcorrection it prevents. Among them:
  - R1: no operand means stock PyTorch, bit for bit.
  - R3: rates are per node and per class of operator, from a functional performance model (Lastovetsky and Reddy,
    2007).
  - R4: counts are derived once per configuration by one deterministic procedure, identical on every rank (MPI 4.1
    §6.7).
  - R5: "Integer rounding minimises the largest finish time: each grain goes to the rank that finishes earliest with
    it" (Beaumont et al., 2001, Alg. 3.1; Ibaraki and Katoh, 1988).
  - R7: the collectives a share waits for are priced inside the derivation, at the largest share (HAP, 2024, §2.4).
  - R8: "Equal counts run the regular collective. Unequal counts run the v-collective with each block at its own
    size, never padded to the largest" (MPI 4.1 §6.5–6.10).
  - R9: what a node can hold is a hard upper bound on its share.
  - R10: nonlinear work is balanced on its own cost function.
  - R14: parts change without shapes changing; each rank's buffer is shaped to the largest share its bounds allow.
- **Sources it names** (its References): heterogeneous ScaLAPACK (Beaumont et al., 2001), AccPar (Song et al.,
  2020), HAP (Zhang et al., 2024), veScale's `RaggedShard` (pytorch#169320), uneven collectives (pytorch#198344),
  HeteroMPI, DFPA, BlueConnect, HexGen, Metis, Whale, and others; 32 in all by the reader's count.

Related documents at HEAD: `design/joint-compilation.md` (how the compilation's optimizers compose),
`design/joint-optimization-sources-2026-11-07.md` (Unity, Alpa, CoCoNet, TACCL, GSPMD, Halide, quoted),
`design/scaling-models-sources-2026-11-07.md` (Amdahl; the α + nβ cost model),
`design/packing-and-coflows-2026-11-06.md` (packing contiguous and non-contiguous pages; co-flows), and
`design/collective-goals.md` (Goal B: "attention heads [7,1] and FFN 7:1").

### Functions

| what | where | at HEAD? | called? |
|---|---|---|---|
| shares that minimise the largest finish time, under bounds, for any nondecreasing cost | `rdma/allocate.py`: `min_max` (67-84), `equal_finish` (47-64), `capacity` (87-88), `Allocator` (5-44) | yes | `rdma/allocate.py` is imported by no tracked file; `min_max`'s `shared` argument has no caller |
| exact adjacent cuts of an extent weighted by shares | `rdma/allocate.py`: `partition_bounds` | history only (be5d8920, 444d4fd9) | — |
| a partition operand on a mesh dimension, and a placement whose shards differ by rank | `rdma/torch-mesh/torch_mesh/partition.py`: `attach`, `split`, `from_local`, `RaggedShard`, `RaggedTensor` | history only (be5d8920; f8fc3950). An untracked build copy remains under `rdma/torch-mesh/build/` | — |
| an input's relation to rotated coordinates with no padding ("Input domains are ragged relations") | `rdma/torch-mesh/torch_mesh/streamed.py`, `structured.py`; metal-microbench `matrix_operations.swift` `butterflyCommands` | history only (mesh 444d4fd9; metal-microbench 3f308655) | — |
| ragged tails lowered through exact partition boundaries | mesh be5d8920; 967ed4ac | history only | — |
| cutting a weight's columns and an attention's heads by shares, and combining the shards' results | metal-microbench `tools/prepare_e2b.py`: `column_cuts`, `packed_cuts`, `partition`, `shard_heads`, `combine`, `kv_pattern` | yes | yes, from `tools/mesh/grid.py` |
| shares per rank as an operand of preparation | metal-microbench `tools/mesh/grid.py` (`--shares`, an even split by default) | yes | yes, from `tools/mesh/run_pair.py` |
| placements with unequal shares | metal-microbench `tools/mesh/placements/attn-7-1-mlp-ledger-bal.json` (attention [7,1], FFN [4660,1484] and [9392,2896]); `even-5.json` (five ranks) | yes | read by `grid.py` |
| sharding in the compiler | `rdma/mesh_compile.py`: `_partition`, `_reshard`, `_partition_pad` | yes | yes; pads to equal shards |

### Measurements

- metal-microbench `docs/tensor_parallel_execution.md`, "Unequal assignment candidates": a table of the best
  complementary cuts between two machines by operation and row count, with the note "The maximum combines separate
  local medians; it is **not measured distributed latency**."
- metal-microbench `docs/hardware_roofline.md`, "3. Use measured, unequal views; tiny shards need repeatability
  checks": "Consequently no single automatically selected partition is installed from these samples." and "Plan
  from shape-specific service curves, clock state and publication cost".
- metal-microbench `docs/measurement.md`, lines 75–86: parts "derived from the link map's proj rates by
  allocate.min_max", and the record of a run whose parts came from another class's rates.
- metal-microbench `docs/amdahl_superiority.md` and `docs/soc_compute_backends.md`: when a split loses to one
  machine, and what each kind of processor offers.

### What the specification names that was not found

- **The link map.** `design/heterogeneity.md` names metal-microbench `configs/links/pair-ring.txt`; `configs/` holds
  `compute`, `draft` and `structure.json` only.
- **A reader of rates.** No function was found that reads a node's rate constants (`proj=`, `attn=`, `mm32=`,
  `sdpa32=`) and builds the cost that `min_max` takes.
- **The price of the collectives at the largest share** (R7). `mesh_collective_time` and `mesh_collective_choose`
  were not found.
- **`tools/torch_parallel/parts.py` and `pair.py`**, which the specification names for the derivation's inputs.
  The directory holds no tracked source.
- **A collective with a size per rank** (R8). None is emitted by the compiler at HEAD.
- **`partition.write` and `partition.sizes`**, named in section 4. The last tracked `partition.py` has `split`.

So for this problem the specification, its sources and the derivation's arithmetic are at HEAD; the operand that
carried unequal shards through torch programs, the ragged relations, and the inputs of the derivation exist only in
history or not at all.

## Part 5. History is judged with the tree

The operator, 2026-11-09:

> then we must judge the relevant history instead of treating this exact snapshot as authoritative in any particular way.

The tracked tree is one snapshot. Two events removed work that answers subproblems of this manifest:

- mesh 1f794b8f (11-02): "torch_mesh is a compiler of four jobs over the run's solve; 21,156 lines of frontend go".
- mesh 0b3e78d7 and metal-microbench 864a0ae4 (11-03): both repositories return to their state of 2026-10-01.
  metal-microbench 8375cb22 (11-03) removes `configs/links/pair-ring.txt` "with mesh_link_map_read".

What follows was found by listing files at commits before those events. Claude read the first lines of a few of
these files and none of their bodies; whether any of it composes with the order of stages is not judged here.

### Capabilities and choices as data

At mesh 90677543 (11-03, the commit before the return), first added at 5dd71ec7 (10-21):

| file | what it holds |
|---|---|
| `etc/compute/apple7.json`, `apple8.json`, `apple9.json`, `apple10.json` | one file per GPU family: its chips, its limits and which features it has, each entry with its source (for example "Apple, Metal Feature Set Tables ... read 2026-10-21") |
| `etc/compute/implementations.json` (289 lines) | implementations by kind: `metal_scalar`, `cpp_scalar`, `metal_engine`, `accelerate_engine`, `ane_engine`, `accelerate_product`, `ane_graph`, `metal_product`, `scalar_contraction` |
| `etc/compute/assignment.json` | `scalar`, `product`, `result_ownership`, `regions`, `collective_cpu` |
| `etc/compute/metal-apis.json` | `scalar_types`, `reduction`, `matrix`, `cpu_matrix`, `graph_operands` |
| `etc/compute/coreml.json`, `solver.json`, `parameters.json`, `streamed.json` | the Core ML model's description; the solver's constants; four named thresholds; the streamed kernels' |
| `etc/compute/metal.def` with `rdma/metal-capabilities.h` | "One manifest, also readable by Swift and the Python extension. No runtime I/O." |
| `rdma/assignment.py` (610 lines), `rdma/mesh_compute.py` (217) | cites `design/whole-function.md`; imports `equal_finish`, `project`, `simplex` from `allocate` |

This is the table Part 2 reported as not found: which processor's supplier offers what, written as data with its
sources.

### The link map and node rates

metal-microbench `configs/links/pair-ring.txt`, at the parent of 8375cb22: link lines "`a b <alpha_us>
<beta_ns_per_byte>`: a link and its alpha-beta crossing cost [Hockney 1994]" and node lines "`node <id> <host>
<gpu-cores> <beacon> proj=<u>,<v> attn=<u>,<v> mm32=<ns> sdpa32=<ns> prim=<ns>,...`", read by `mesh_link_map_read`
and `mesh_link_table_state`. This is the link map and the rates that `design/heterogeneity.md` specifies and that
Part 4 reported as not found.

### Unequal shards and ragged relations in torch programs

`rdma/torch-mesh/torch_mesh/`, by commit:

| commit | date | `partition.py` | `streamed.py` |
|---|---|---:|---:|
| f8fc3950 | 10-23 | 1,289 lines | 1,795 |
| be5d8920 | 10-28 | 1,836 | 1,996 |
| 444d4fd9 | 10-29 | 1,843 | 2,025 |
| parent of 1f794b8f | 11-02 | 1,482 | 2,085 |
| 90677543 | 11-03 | 78 | absent |

At the parent of 1f794b8f the package also held `native.py` (2,427 lines), `declarations.py` (1,716),
`streamed.metal` (1,521), `engines.py` (1,176), `ownership.py` (1,151), `page_effects.py` (775), `compilation.py`
(767), `operand.py` (687), `attention.metal` (613), `placement.py` (466), `algebra.py` (413), `structured.py` (326),
`collective_views.py` (272), `layout.py` (258) and others. Part 4 names the functions a reader read at be5d8920 and
444d4fd9 (`RaggedShard`, `attach`, `split`, `from_local`, `partition_bounds`, the ragged relation of `streamed.py`).

### Other modules present before the return and absent now

In mesh `rdma/` at 90677543: `mesh_relation.py` (532 lines), `mesh_matrix.py` (313), `mesh_sparse.py` (61),
`allocate.py` (208 lines there; 88 at HEAD), `mesh-page-map.c` (344), `mesh-routed.c` (430), `mesh-fabric.c`,
`mesh-plan.h`, `mesh-span-plan.h`, `mesh-page-lower.h`, `mesh-publication.h`, `mesh-write-compiler.h`,
`mesh-cpu-source.m`, `mesh.metal`, `isl.patch`. In all, 156 files outside `xonotic/` and `measurements/` were present
at 90677543 and are absent at HEAD.

### What this means for the manifest

- The two rows Part 2 called not found, and the six items Part 4 listed as named by the specification and not found, are each
  either in history as shown above or still not found (the price of collectives at the largest share; a collective
  with a size per rank).
- Being in history decides nothing about direction of data flow. The same test applies to a function in history as
  to one at HEAD: its type at its boundary against the order of stages.
- This part lists; it does not survey. The commits between 2026-10-01 and 2026-11-03 were not read.

