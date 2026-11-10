# Change requests: which are outstanding

As of 2026-11-09, including the composition implementation and review recorded in metal-microbench
`output_data/claude-ir-handoffs-20261109T051956Z` and the continuing replacement workgroup in
`output_data/claude-typed-replacement-20261109T085408Z`. Each entry names a change request, whether it is
outstanding, and what of it is outstanding. The standard every change request follows is
[change-requests.md](change-requests.md).

## Outstanding

- [compilation-composition-change-request.md](compilation-composition-change-request.md): outstanding, items 1–7.
  Composes the nine requests below through cited compiler roles, complete lowering, shared representations and
  module ownership. Their individual implementation records do not establish that this composition is complete.
  Item 5 records the operator's native Apple Silicon execution and implementation deduplication correction and
  assigns replacement of duplicated specialization, command orchestration and representation bodies.
  Item 6 requires completing IR handoffs before helper extraction. The
  [complete replacement workgroup](compilation-composition-change-request.md#complete-replacement-workgroup)
  assigns eight connected responsibilities and one coordinated replacement landing across both repositories.
  Working commits record development; they do not constitute that landing or narrow the assigned program domain.
  [Item 7](compilation-composition-change-request.md#7-preserve-the-properties-while-compiling-away-the-abstract-machinery)
  cites MLIR analysis/control lowering, Halide scheduled program synthesis and CompCert behavioral preservation.
  Lowerable IR carries solved properties into optimizing compilation; the runtime need not retain their original
  types/graphs/metadata. Fixed target operations in the scheduled order must displace the remaining task/loop
  and command interpreters. This documentation records the obligation, not completion or a performance result.

  Current connected path: source-built LiteRT/ML Drift and Torch numerical producers → owned typed numerical
  operations and Tint regions → specialization, fusion and target lowering → retained effects, dependence causes,
  placement and native execution. Generated-source interception, parsing/rewrite bodies, native source recognizers
  and the non-self-timed preparation bypass are deleted. Graph algorithms remain in `mesh-graph`/`mesh_graph`.
  Selected tuned constructions feed one member emitter in normal and later kernels; the temporary blanket
  no-fusion gate and repeated native selection are gone. CPU/GPU preparation shares scheduling and initialization;
  replay ownership survives native row expansion. A10's call concurrency is separate from retained-storage horizon.

  Compiler, prepared/native and collective-plan artifacts retain solved results and relocate current storage on
  load. Existing restored numerical cases pass at metal-microbench a136b464 / mesh435bc6ae: normalization/projection
  relative RMS0.000216, matrix projection0.000207, CPU projection0.000276, argmax zero mismatches, all outputs written
  and finite. CoreML restoration also has numerical evidence, but its measured plan was ANE0/CPU13/GPU0; this does
  not establish Neural Engine execution. The composed Torch save/load consumer still needs operational evidence.

  The production dependency chain builds. All five E2B stage graphs publish/lower/finalize in the latest compile-only
  record (build26): 3/7/20/602/1068 dispatches. An earlier decode finalization crash in AGX does not reproduce after the changed
  member emission; its cause is unproven. These are compilation records, not full-model numerical or throughput
  evidence. Commands, revisions, binary hashes, observations and superseded development findings are retained in
  metal-microbench `output_data/claude-typed-replacement-20261109T085408Z`.

  The shared typed LU/triangular/Cholesky, Householder QR, Padé matrix exponential, Hermitian eigendecomposition and
  one-sided SVD producers now have limited runtime evidence through the existing native artifact diagnostic.
  At metal2f845c87 / mesh9186b1f8, restored CPU inverse has relative RMS3.942e-8 and the matrix exponential2.667e-6;
  each numerical region compiles to one native CPU task, including the exponential's loop. The inverse retains a
  separate typed CHECK operation. Restored GPU inverse and matrix exponential have the same accuracy; full real3x2
  QR has reconstruction error7.006e-8 and orthogonality error1.681e-7. Singular CPU inverse reports typed info2/EDOM
  instead of numerical success. Public Torch exception translation still needs operational evidence.
  At metaldbfd53dd / mesh69a971ce, restored native CPU complex Hermitian eigendecomposition has reconstruction
  error1.269e-7; real full SVD2.799e-8 and rank-deficient complex full SVD1.701e-8, with orthogonality errors below
  1.8e-7. Restored I64/F64/BOOL sort and gather preserve exact values and stable indices, including wide values;
  direct binary64 FMA preserves single rounding, overflow cancellation, subnormals and special values in eight
  concrete lanes. A zero-operation graph preserves transposed/sliced input aliases and a constant across restore.
  At metalfb8c3ffa / meshf7b77d43, GPU eigendecomposition and both SVD cases execute after retained IR handles,
  unique-target compilation and compiler/archive ownership changes; the earlier AGX crash does not reproduce.
  GPU complex Hermitian reconstruction error is4.284e-8; the SVD errors match the CPU cases above. This does not
  isolate which change removed the driver failure. CPU determinant/slogdet/pseudoinverse composition also executes
  for a singular2x2, preserving the expected nonfinite log determinant and pseudoinverse error1.133e-8.
  At metal012f52bd / meshc4fbcac8, native CPU LDL factor/solve executes a batch containing a2x2 pivot and nontrivial
  swap, with solve relative error3.274e-8 and residual3.377e-8. Mesh-produced factors work in Torch's solve, and the
  compiled solve consumes Torch-produced factors. These cases do not establish all shapes, convergence/extreme
  scales or performance. At metalf91334a9 / meshcdf04321, completing canonical storage bindings, typed gather/local
  handoffs and whole-control local storage reduces the tiny GPU eig from154 dispatch records to5 (4 GPU commands
  and a typed CHECK). Its restored artifact falls from12,340,083 to420,912 bytes; eigen residual2.504e-8 remains
  correct. One Program-owned effect/carriedness cache now supplies transformation and retention. This is structural
  and numerical evidence, not measured throughput. Build19 (metalc5e70a2f / meshc626df13) retains compatible CPU
  member groups and their locals; the 81-operation matrix exponential restores as one native task, with relative
  error 2.667e-6 and a 192,895-byte artifact. GPU eig still has five dispatch records and residual 3.541e-8.
  General eigenvalues now compose Householder Hessenberg reduction, implicit shifted QR and shared triangular
  substitution. The first rotation case exposed an in-place complex component-order defect: exit0 accompanied
  a56.802 residual. Yielding both typed components in one logical iteration corrects the shared producer; the
  rotation residual is6.384e-8 and full complex QR reconstruction1.082e-7. Mesh a091ae92 also applies paired
  results to contractions and adds pre-contraction scaling. Build18 executes these changes through CPU rotation
  (residual6.384e-8) and GPU full complex QR (reconstruction8.080e-8, orthogonality2.574e-7,2 dispatches).
  Mesh5b61507c adds fixed-output least-squares driver composition through the existing QR/LQ, pivoted QR/RZ and
  SVD lowerings, with original driver and output-layout metadata. Build19 same-driver solution errors are 3.589e-7
  for native CPU tall GELS, 4.300e-7 for GPU wide complex GELS and 9.644e-8 for native CPU rank-deficient GELSY;
  residual/rank fields agree. Build19 GPU wide GELSD had a close solution but incorrectly reported rank two instead
  of one. Compiler review found fusion typing a physical F32 input from a later BOOL semantic parameter after
  composing a comparison prologue. Metal0204b00b corrects both fusion interfaces to consume the physical input's
  reader type. A fresh build20 GPU artifact restores with rank one and solution error 1.287e-7; fusion stays enabled.
  Metal9dc15107 supplies eighteen authored binary64 elementary functions through SoftFloat pairs and sourced
  FDlibm range reduction and approximations. Build19 exports, compiles and restores these plus shared log1p/expm1
  as one GPU dispatch. A 32-lane extreme/special-value vector differs from Torch CPU by at most two ULPs, including
  huge trigonometric arguments through maxdouble; the tiny negative log1p result agrees with libm where Torch
  rounds it to zero. These finite examples do not establish a full-domain accuracy bound. The restored training
  and collective callers now use upstream joint forward/backward export and current public program/save/load.
  Build20 executes the canonical transformer's retained joint graph through saved native artifacts: one layer,
  width eight, two heads, four sequence positions, nine parameter gradients and an input gradient. Its 119 typed
  operations become five GPU dispatches or one native CPU task. Forward relative errors are 8.270e-8 and 7.236e-8;
  all gradient errors are below 2.020e-7 and the loss error is 2.362e-7. The artifacts are 521,972 and 378,827 bytes.
  This establishes those joint numerical paths and restoration, not distributed public-program execution.

  Remaining work: complete frontend/target coverage including binary64 functions; justified mixed-target handoffs; paged CPU/write-ahead and compiled execution-index behavior; complete sourced
  actor costs and whole-group decisions; typed dynamic synchronization observation; actual Torch/collective and
  five-node E2B execution with the required transport overhead/bandwidth and decode throughput after prefill.
  Source review also identifies specific performance mechanisms still to preserve: typed native CPU SGEMM/BNNS
  contractions replacing the old Accelerate path; submission of independent GPU work before synchronous CPU/ANE
  computation where the compiled schedule requires overlap; and efficient typed indexed updates. Build22 connects
  typed BLAS/BNNS calls inside the existing native member body, with retained layout/workspace and shared contraction
  facts consumed by CPU, GPU and Core ML. Restored F64 transposed and F32 single-row BLAS cases agree exactly with
  Torch. Build23 verifies corrective checkpoint metalaa0dc393: strided BNNS, fixed-half staging and
  rounded-input contractions restore with exact agreement. The canonical joint transformer also executes through
  native CPU after tuned contraction lowering: forward relative error 7.737e-8, parameter gradients at most
  2.878e-7, input gradient 1.403e-7, all finite; the 119-operation artifact is 413,734 bytes. These results resolve
  the build22 printer/binding failures and preserve those numerical cases; throughput remains unmeasured.
  The output-by-update scan is replaced by stable key sorting, shared segmented scans and canonical typed scatter.
  The shared lowerer now suppresses ordinary mapped writes for explicitly scattered outputs. Build22 GPU execution
  covers all eleven update variants, duplicate keys, include-self/alpha, I64/complex128/Bool values, noncontiguous
  advanced indices, scalar/empty cases and histogram. The largest relative difference is 2.558e-8; the other checked
  results are exact. Alpha is applied before reduction, preserving the finite 1.5e38 overflow-sensitive case.
  Invalid negative scatter indices and an index-add coordinate of 2**32 report the intended typed check. CPU update
  execution passes in build23 after the shared printer correction: ten of eleven update results are exact,
  scatter_add differs by 2.558e-8, and complex128 and alpha-overflow cases are exact. This does not establish performance or
  distributed EP execution. Restored CP/EP/mixed-engine callers do not by themselves close those obligations.
  Compiler checkpoint metal2abd0b2f moves fixed Core ML feature providers, array views, options and output bindings
  to prepared invocation objects, preserving the existing `CoreMLFunction` preparation boundary. The adjacent
  executor binds relocated storage once per residue. Build24 restores this handoff with correct numerical output.
  Both the existing F32 arithmetic case and an explicit historical F16 contraction still receive CPU-preferred
  plans. Review found target consumers bypassing constant bytes owned by the Value's storage root; that handoff
  and the historical constant-weight convolution construction remain under repair. A Core ML preferred-device
  plan reports intended placement, not observed device use.
  Build24 also executes a mixed GPU/CPU/GPU artifact with relative error 4.298e-9, but its signature has one false
  edge: the new prefix submission is attributed to its numeric consumer instead of the actual GPU producer.
  Build25 repairs the retained provenance and passes external observation, with one image edge and no false edge.
  GPU-prefix events and whole-call completion retain their distinct owners; overlap throughput remains unmeasured.
  The historical fragment-1 input now has a typed codec producer (mesha743225e), retained selected/basis operands,
  and explicit source-device versus compiled-target placement. Its first nonzero numerical cases fail on CPU and
  GPU because the handoff equation included row parity twice. Mesh13d6d475 corrects that one inverse lane formula
  against the historical packer. Build25 passes unchanged-reference numerical checks: plain CPU/GPU, int8 rotation,
  Hadamard and ragged FFN all have relative error at most 1.828e-7; zero-width output is exact. A separate signed I4
  strided/offset view over constant root storage also restores exactly. Target-demanded compilation of invariant
  coded definitions remains open. Build26 verifies selected packed projection (algorithm1) for a signed I4 view
  with offset5 and strides[1033,2], with exact output. The repaired F16 neural convolution executes exactly at
  8x128x64 and 1x2048x1024 for the supplied power-of-two inputs, but both preferred plans still select CPU.
  Build27 replaces the native scalar constant evaluator with one compiled preparation task. Build28 retains
  demanded whole producer definitions, orders them through existing member order/shared graph traversal, and saves
  materialization provenance in artifact v20. CPU plain/rotation/Hadamard/zero/ragged coded cases preserve their
  numerical results. Review identifies missing immutable-source annotations/bytes before lowering: the production
  path assigned all planned invariants STATE, while the diagnostic supplied attribute bytes only as later input
  files. Mesh5942f839 carries export constants and explicitly immutable coded definitions through the same initial
  storage owner before lowering, excluding captured mutation and compiled writes. Ordinary mutable parameters and
  buffers retain state semantics. Fresh CPU/GPU preservation results pass against unchanged references, but
  plain/zero/ragged neural compilation still fails with region access other than a struct member. Exact typed
  inspection identifies operation0 as the dynamic input permutation gather, preceding operation1's constant
  decoder. The remaining failure is in Core ML's region-access consumer; it cannot establish whether decoder
  preparation succeeded. That consumer and the independent materialization evidence remain assigned.
  Both F16 literal-weight cases remain exact but CPU-preferred, including the convolution itself. Build28 removes
  the duplicated preparation payload: the larger artifact falls from8,410,794 to4,204,077bytes. The zero-operation
  alias/constant diagnostic also remains exact with its existing externally supplied literal initialization;
  that does not establish the still-pending public outer save/load path. Target preparation remains under revision.
  A further F16 R128/K3840/N4096 artifact restores exactly for its supplied power-of-two inputs, and all eleven
  nonconstant MIL operations, including convolution, are Neural Engine-preferred. One subsequent existing
  native_bench saved/restore invocation under Instruments records an actual ANE inference prediction, with output
  bytes identical to the earlier numerical result. This establishes ANE execution for that geometry; no historical
  row-tiling throughput comparison or broader target-coverage closure follows. Online KV attention,
  local result layouts and global dropout coordinates now have source checkpoints. Build21 restores a two-block
  grouped-query attention case with query shape [1,4,3,128], KV length 35 and value width 96. Masked attention has
  relative error 2.217e-7, including an exactly zero fully masked row; causal attention has error 6.092e-8. The two
  operations become 22 typed operations and a 454,799-byte artifact. This does not establish distributed or dropout
  execution. Build21 also passes fresh publication/lowering/finalization of all five E2B stages after extending the
  retained resource/control cost DAG; dispatch projections share one DAG evaluation per supplied machine.
  Missing machine operands remain explicit in cost estimates. No corresponding operational/performance closure is
  recorded. Claude quota exhausted during this workgroup; the operator-authorized Codex continuation remains active.
- [synchronization-signature-change-request.md](synchronization-signature-change-request.md): outstanding.
  The graph schema, serialization and printers, plan/native emission, composition and external monitor are present
  in the composition implementation. The review records distinguish remaining representation and observation gaps
  from implemented source. The existing external monitor observes the restored CPU projection's one expected
  host signal at mesh 53bc3b17, with numerical agreement. The restored GPU projection also passes external
  observation of five program operations; its three kernel records are static compiler data. Bridge observation
  and fail-stop interception remain open. Typed invocation spans are emitted and consumed. At metalcc3618ff /
  mesh69a971ce, three-span argmax passes observation of17 program operations, with9 static kernel records,2 image
  edges and no redundant edges. Retained allocator-completion facts remove the GPU fences that duplicated its
  required completion waits; argmax numerical mismatch counts remain0.
  At metal012f52bd, the GPU eig's CHECK wait consumes the same commit-completion fact as allocator reuse, removing
  its two redundant synchronization edges. External observation initially exceeded macOS's total DOF load limit;
  smaller initializer chunks and short clauses alone did not fix it. At mesh ba80ddcc, grouping equivalent
  compatible-state transitions into bit sets enables and observes all161 program operations successfully;
  its215 kernel records remain static IR facts. Build18's fused eig also passes external observation:13 program
  operations,7 static kernel records,1 image edge and no redundant edges.
  A deliberate mismatched existing CPU client/signature could not exercise fail-stop: macOS denied destructive
  DTrace before client execution. No stopped client remained and no machine security configuration was changed.
  Build19 retains submission actions and binds their meaning to the actual recorder image UUID. Fused eig passes
  external observation of 12 program operations and seven static kernel records, with one image edge and no
  redundant edges. At build27 (metal63c465ab / mesh22f409d0), mixed GPU/CPU/GPU observes10 operations and one image
  edge; three-span argmax observes17 operations and two image edges, with zero numerical mismatches; neither has a
  false or redundant synchronization edge. Event-role selection uses the configured identity after composition,
  and the optimized runtime gains no provenance bookkeeping. These runs use --observe and do not exercise
  destructive fail-stop interception. Metalcc580af4 / mesh8ce87a05 also retain one exact conflict result, including
  self conflicts, consumed by ordinary dependencies. Artifact v18 retains it on save/load.
  These limited observations do not close items1–6.
  Struck 2026-11-09 (compilation-composition-change-request.md, "Withdrawn, 2026-11-09"): this entry's record of
  builds 20 to 27 observing conditionals, nested regions and WHILE loops, and the work it listed as remaining for
  them. None of it is outstanding.
- [torch-capture-change-request.md](torch-capture-change-request.md): outstanding in part.
  - Item 3 in the LiteRT scope: implemented in metal-microbench 5600d84b, 66a18fab, c0a067fc and 6b8f7c3c, and mesh
    56e33484, e34e7a5e and b482a825. A received operand's reader is masked, the threadgroups whose pages have not
    landed are compacted into a worklist and run by a later pass's indirect dispatch, and the later passes are placed
    by the heap order over the commands' dependence levels and their ranges' published completion times. Checked on
    this Mac by the scratch mask harness, and by the five-rank E2B decode on the five machines, 71 later passes a step
    (integration-finishing item 1).
  - Open implementation: item 2's execution-time estimates t(v), on which its buffer bound depends. The
    2026-11-09 appendix supplies the analytical cost-model source and its connection to the existing buffer-bound
    method. The replacement connects dispatch actor records to `mesh_graph_actor_time`, `mesh_plan`'s existing
    buffer-bound calculation and the native program's horizon; `HOST_LAG` is removed. Target execution
    multiplicities, machine operand meanings/coverage and mixed CPU/GPU actor topology still require revision;
    zero estimates from absent records do not establish the required storage or performance behavior.
  - In the torch scope, item 9 remains outstanding across the full assigned domain. The composition implementation
    removes `program()`'s MPS-only/no-transfer selection and connects CPU tasks, MPS replay and declared transfers
    through the native program. Its cross-placement results carry source-call readiness to their consumers.
    Full coverage and operational evidence remain unfinished. The lowering request's withdrawn item 4 points to
    item 1 for the consumer's lowering; its withdrawal does not complete or withdraw item 9.
  - Implemented: items 1 (metal-microbench a3792f13), 4 (mesh 992f6389), 5, 6 (mesh a2c1532c and b880b808,
    metal-microbench 6b3c29e5, completed in mesh 356bed56) and 7, and item 2's first write. Item 8 is written by the lowering
    request's items 1–3.
- [problems-of-14b9a757-change-request.md](problems-of-14b9a757-change-request.md): outstanding in part.
  - Implemented: the requirements written from its stops in mesh 593dd1c0. These are the second stops' item 8 and the
    third stops' "loading only the slices a node reads", by Megatron-LM's `_initialize_affine_weight_cpu`, and the
    third stops' collective over a view of an invariant, by GSPMD §3.1 and §3.5. They are in mesh 2155c2b8 and
    60f43ddb, completed in 76d26e3b.
  - Settled: the second stops' u copies, by its appendix item 14. Superseded: the first stops' item 8.
  - Problem 4: mesh 30afa696 implements the replicated window without halo exchange and the middle's local copy
    where needed; the former blanket statement that no implementation commit exists was stale. The 2026-11-09
    workgroup review also finds the appendix's multi-spatial-dimension composition in the current source:
    `_partition` visits dimensions in increasing order and gives each `_region` the previous region's result;
    its halo slices therefore span the earlier dimension's halos and carry the corners. View-to-view transfers
    consume those shapes, one-shard dimensions are skipped, and replicated windows remain local. The clause-by-clause
    source record is the replacement run's `6_effects.status.md`; no two-dimensional sharded-window operational run
    is recorded. This corrects the code-status gap without claiming that execution evidence.
  - Problem 11: mesh 028b27d6 removes cancelled plans before replacement children can load them, with an acquire
    read of `removed` in `plans_load`; that implementation remains in the current source. The former no-commit
    statement was stale. This structural review did not perform a link-loss workload.
- [plan-to-bridge-change-request.md](plan-to-bridge-change-request.md): outstanding in part.
  - R4, "Behaviour at run time" (added 2026-11-09): outstanding. Four statements of what a run does for a function
    that needs its operand whole and for one that takes pages in any order. Ordered in
    implementation-order-2026-11-09.md under items 4, 7 and 11. No run with a received operand exists at the current
    tree, and no quoted source says which record of a run shows them.
  - R4's consumption of a received operand one partition at a time has historical implementations in the torch
    scope through the lowering request's item 1 and in LiteRT through torch-capture item 3. In the replacement,
    build19 carries per-group footprints/page masks into native first-pass, compaction and deferred work, and
    retains a shared CPU/GPU dispatch order. Received-operand runtime evidence
    remains pending. Build20 restores the receive-completion priority through
    compiler-owned legal insertion bounds and a final permutation over the retained relation. Recorder lifetimes
    use stable preparation anchors, extending every movable pass operand over its permitted interval; normal load
    restores the permutation without another solve. Metale1791a34 also persists last-use handover records and orders
    a shared-cut handover after all deferred passes at that cut. Transport/overlap evidence remains pending, so these
    source changes do not close R4.

- [unequal-shards-change-request.md](unequal-shards-change-request.md): outstanding, items 1–15. Written 2026-11-09
  from `design/heterogeneity.md`; nothing of it is implemented at mesh 7df87fe4. Written since, 2026-11-10, in
  `rdma/mesh_compile.py` unless another file is named:
  - Items 1, 15 and 3: `Shard` (an entry of `splits[name]`: the mesh dimension with its tuple), `_sizes` and
    `IR.counts`, read by a slice and by `IR.local`; an all-gather, an all-to-all and a reduce-scatter carry the counts
    (`Collective.counts`) and rank j's block is `counts[j]` elements, placed after the blocks of the ranks before it.
  - Items 2, 12, 9 and 10: a rank's buffer is the `max(counts)` elements of the dimension from its base and its shard
    the subarray at `starts[rank]` (`IR.bases`, `IR.starts`); `Collective` carries `counts`, `displs` and `starts`, and
    each lowering sends and places the subarray; the slice of a replicated dimension is a view at the rank's base;
    `_partition_pad`, the `pad` item and the `narrow` item are deleted; `_region` exchanges the halo from the rank j
    places away by one collective permute at the largest size over the ranks, and a rank copies its own part where a
    sender's shard is smaller than that size. Where a resharding keeps the sharded dimensions and their counts and
    changes their mesh dimensions, `_reshard` emits one collective permute (XLA's `ReshardWithCollectivePermute`), its
    record carrying each place's counts, displacements and starts, and `_collective_permute` sends and places the
    subarray.
  - Item 4: before a call that contracts a dimension whose ranks' shards differ, each operand holding the dimension is
    written through `_shard_select`, a select on the index along the dimension compared with the rank's `starts` and
    `counts`, 0 outside the shard, at a rank whose shard is smaller than its buffer (`IR.masks`). The select is its own
    operation writing a temporary; it is not part of the contraction's region. The narrowing of a contraction's
    operands (`IR.trims`) is deleted.
  - Item 5: `examples/tp.py` and `examples/pl.py` `placement` cut the feed-forward block's first weights along
    dimension 0 and its second along dimension 1 by one entry, `Shard(0, units)` where the environment gives `UNITS`;
    `_partition` then contracts each rank's shard and emits one reduction, with no collective before the second
    product. A linear with a bias is contracted at each rank without the bias, and the bias is added to the reduced
    output once, by a call the partition adds after the collective. metal-microbench `tools/prepare_e2b.py`
    `column_cuts` accumulates item 1's sizes into its cuts, with the packed-column alignment after; `partition` is
    kept as it is.
  - Items 6 and 14: `rdma/mesh-graph.h` `mesh_graph_shares` (`rdma/mesh_graph.py` `shares`) solves HAP's program (3)
    with HiGHS: the shares minimise, summed over the stages, the stage's collective time a unit of the largest share
    times the largest share, plus the largest over ranks of the rank's computation time, a fixed part and a part a unit
    of its share; the shares sum to 1 and each lies between 0 and the rank's upper bound, the variable's bound.
    `shard(m, extent, comm, comp, upper)` gives one part's placement, its tuple the whole sizes of those shares by
    `_sizes`. `examples/tp.py` and `examples/pl.py` `placement` take `parts`, the stages of each feed-forward block by
    its name, and call `shard` for each; the one tuple for every block (`UNITS`) is deleted, and `PARTS` gives the
    stages as JSON. `rdma/allocate.py` `min_max`'s `shared` argument is deleted. Where shares are given,
    `rdma/allocate.py` `shift` is Whale's Algorithm 1.
  - Item 7: `rdma/allocate.py` `speed(times, x)` is x / t(x) from one node's measured times of one kind of function.

  Outstanding:
  - Item 2's gather and scatter. No quoted sentence says where the program emits either: every input and constant is
    held at every node, a resharding replicates on every device (GSPMD §4.5), and the one transfer from a single
    device is `_deliver`'s, of a whole output.
  - Item 12's other form, whole padded buffers through an equal-size collective, and its choice (item 6, which decides
    it, is not written).
  - Items 4 and 5: a run. None was made of either; each was partitioned and lowered in process, rank by rank.
  - Item 5: a convolution with a bias (`_convolution`, `Rule.fused`) still has its input channels gathered before the
    product.
  - Items 6 and 7: no measured time of any function on any node is in either repository, and no quoted sentence says
    at which x a rank's speed s(x) is taken to give (3) its linear computation time, or which of a program's
    collectives make a stage and what its time is a unit of the largest share. `speed` has no caller, and a client's
    `parts` come from its caller.
  - Item 14: `shift` has no caller; no client is given shares with what each rank can hold.
  - Items 6, 7 and 14: a run. None was made; `tp.py`'s feed-forward blocks were partitioned in process with a tuple
    solved for each.
  - Items 8, 11 and 13.

- [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md), item 5 (2026-11-09): outstanding.
  Mesh 199eb9c3 deleted the lowerings of matrix exponential, Hermitian and general eigendecomposition, SVD,
  pseudoinverse and the gelss and gelsd least-squares drivers, which repeated on a tensor predicate. What this
  index says above of their numerical evidence describes code that no longer exists. Item 5 states what replaces
  them.

## Implemented

- [integration-finishing-change-request.md](integration-finishing-change-request.md): implemented.
  1. Every bridge is configured as the repository installs it and runs. The job io.mesh.bridge was disabled on all
     five machines and is enabled; the mini's stale /usr/local/mesh/bridge.conf (2026-09-17) is removed; each
     machine's ~/.mesh-bridge.conf states region=/mesh0, its arena (82 GiB on the Ultras, 16 GiB on the mini, as the
     deleted etc/bridge profiles set them) and mesh_block_pages=1, and no window, so the window is mesh-bridge.sh's
     default, a bank's pages; the mini's configuration.plist is rewritten by the same start. The window was refused
     on every port at the provider's advertised max_mr_size; device_up registers it in 1 GiB regions as M09 states
     (mesh b59c579a, e6eda244), four a port, and all five maps list all five nodes. The five-rank E2B decode
     (metal-microbench tools/mesh/placements/even-5.json, 03298a79) runs to its end on all five machines: in
     output_data/e2b-five-2026-11-08 and its three repeats in output_data/e2b-five-2026-11-08-repeats, every rank's
     prefill and decode end, every rank exits 0 and decodes the same 256 tokens, 33.7 tok/s (29.7 ms a step, the
     median of three).
  2. Implemented in metal-microbench c0a067fc and 6b8f7c3c with mesh b482a825.
  3. Implemented in mesh 01632014.
  4. Implemented in mesh e34e7a5e and metal-microbench c0a067fc, which C9 passes for MetalMaskedPass.
  5. Implemented in metal-microbench 5d5efc09.
  6. This index.
  7. Implemented in metal-microbench 32cda1e1: one MTL4CommitOptions a commit, its handler added for that commit, in
     tools/e2b_replay.m and metal_recording.m; with mesh 792cbb7c and metal-microbench faa18aba for C9's finding on
     32cda1e1. Every rank's prefill timeout of 2026-11-08 does not recur.
- [torch-source-lowering-change-request.md](torch-source-lowering-change-request.md): implementations in the cited
  revisions were checked on this Mac and built on the five mesh machines. Full-domain completion and preservation
  of typed annotations through the whole program remain under compilation-composition; the historical entries
  below do not close those requirements.
  1. Each function typed and scheduled as plan-to-bridge R3 and R4 write it, on every Apple silicon backend, the CPU
     included (rewritten 2026-11-08). Implemented in mesh 502bf42c and 8cac6d69 and metal-microbench 3bc0e2a4: the IR's
     kinds, one task per page for the asynchronous consumer, the same source compiled for the CPU, and A3's weighting
     operand, a weight on each of mesh_plan's tasks and on each edge (mesh 01632014). Reductions, softmax and attention
     as R4's homomorphisms, one task per page combined by their sources' ⊕ (design/homomorphism-sources-2026-11-08.md),
     and a move between backends split into functions ordered by an event: mesh 2abc7125, metal-microbench 71a23822.
     The all-reduce root's and reduce-scatter's sum of received partials, one task per received page, in the order of
     their published completion times: mesh 3e176a9d, a582ce3c and 91786869.
  2. The capture is a static dataflow graph. Implemented in mesh b4a6e67d.
  3. Self-timed lowering of dependences into Metal orderings. Implemented in mesh 38e2b7e6 and 775f142c and
     metal-microbench e4800f65, 270c6e5f and 43f37e96. The order is plan_schedule's: its completion times are
     published with the plan (mesh a582ce3c), the second linear program over the GPU's commands is deleted
     (metal-microbench 0a910421), and the replay's list is the heap order over those keys (integration-finishing
     item 2).
  4. Withdrawn 2026-11-08: plan-to-bridge R4 writes this consumer, and item 1 lowers it. Its code is deleted
     (metal-microbench 5d5efc09).
- [aot-compilation-change-request.md](aot-compilation-change-request.md): A1–A10 have implementations, A3's operand as
  integration-finishing item 3 writes it (mesh 01632014), and its appendix's departures were addressed in commits of
  2026-11-05. Appendix item 7, the torch path on torch's existing kernels, is
  superseded by the lowering request's item 1.
  These are historical implementation records, not closure of the replacement's A1–A10 obligations. In particular,
  A1's saved/resumed executable now has compiler, prepared/native and collective-plan artifact producers and
  consumers in the continuing replacement. CPU, GPU and CoreML target restoration have numerical evidence;
  the composed Torch save/load consumer has only compile-time review.
  Restoring solved results and relocating current storage is distinct from running analysis again. Full coverage
  and operational evidence for that handoff remain assigned to the representation, Torch and executor owners.
  The composition review's remaining representation, source-domain and execution-path findings still apply.

## Superseded or closed

- [disjoint-pages-change-request.md](disjoint-pages-change-request.md): D.2, D.3.1 and D.5 are superseded by the
  plan-to-bridge request; the rest is implemented.
- [page-queue-change-request.md](page-queue-change-request.md): superseded in part; the rest is implemented.
