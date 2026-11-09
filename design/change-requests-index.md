# Change requests: which are outstanding

As of 2026-11-09, including the composition implementation and review recorded in metal-microbench
`output_data/claude-ir-handoffs-20261109T051956Z` and the continuing replacement workgroup in
`output_data/claude-typed-replacement-20261109T085408Z`. Each entry names a change request, whether it is
outstanding, and what of it is outstanding. The standard every change request follows is
[change-requests.md](change-requests.md).

## Outstanding

- [compilation-composition-change-request.md](compilation-composition-change-request.md): outstanding, items 1–6.
  Composes the nine requests below through cited compiler roles, complete lowering, shared representations and
  module ownership. Their individual implementation records do not establish that this composition is complete.
  Item 5 records the operator's native Apple Silicon execution and implementation deduplication correction and
  assigns replacement of duplicated specialization, command orchestration and representation bodies.
  Item 6 requires completing IR handoffs before helper extraction. The
  [complete replacement workgroup](compilation-composition-change-request.md#complete-replacement-workgroup)
  assigns eight connected responsibilities and one coordinated replacement landing across both repositories.
  Working commits record development; they do not constitute that landing or narrow the assigned program domain.

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
  record: 3/7/20/618/1068 dispatches. An earlier decode finalization crash in AGX does not reproduce after the changed
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
  and numerical evidence, not measured throughput. CPU composition still requires compatible-grid local fusion.
  General eigenvalues now compose Householder Hessenberg reduction, implicit shifted QR and shared triangular
  substitution. The first rotation case exposed an in-place complex component-order defect: exit0 accompanied
  a56.802 residual. Yielding both typed components in one logical iteration corrects the shared producer; the
  rotation residual is6.384e-8 and full complex QR reconstruction1.082e-7. Mesh a091ae92 also applies paired
  results to contractions and adds pre-contraction scaling. Build18 executes these changes through CPU rotation
  (residual6.384e-8) and GPU full complex QR (reconstruction8.080e-8, orthogonality2.574e-7,2 dispatches).
  Mesh5b61507c adds fixed-output least-squares driver composition through the existing QR/LQ, pivoted QR/RZ and
  SVD lowerings, with original driver and output-layout metadata. Its same-driver numerical checks remain pending.
  Metal9dc15107 supplies eighteen authored binary64 elementary functions through SoftFloat pairs and sourced
  FDlibm range reduction and approximations. Tint reads that library into IR; connected target compilation and
  numerical range/ULP evidence are pending. Neither source checkpoint closes the assigned numerical domain.

  Remaining work: complete frontend/target coverage including binary64 functions; target-owned structured control
  and justified mixed-target handoffs; paged CPU/write-ahead and compiled execution-index behavior; complete sourced
  actor costs and whole-group decisions; typed dynamic synchronization observation; actual Torch/collective and
  five-node E2B execution with the required transport overhead/bandwidth and decode throughput after prefill.
  Missing machine operands remain explicit in cost estimates. No corresponding operational/performance closure is
  recorded. Claude quota exhausted during this workgroup; the operator-authorized Codex continuation remains active.
- [synchronization-signature-change-request.md](synchronization-signature-change-request.md): outstanding.
  The graph schema, serialization and printers, plan/native emission, composition and external monitor are present
  in the composition implementation. The review records distinguish remaining representation and observation gaps
  from implemented source. The existing external monitor observes the restored CPU projection's one expected
  host signal at mesh 53bc3b17, with numerical agreement. The restored GPU projection also passes external
  observation of five program operations; its three kernel records are static compiler data. Bridge and
  structured-control observation and fail-stop interception remain open. Typed invocation spans are emitted and
  consumed. At metalcc3618ff / mesh69a971ce, three-span argmax passes observation of17 program operations, with9
  static kernel records,2 image edges and no redundant edges. Retained allocator-completion facts remove the
  GPU fences that duplicated its required completion waits; argmax numerical mismatch counts remain0.
  At metal012f52bd, the GPU eig's CHECK wait consumes the same commit-completion fact as allocator reuse, removing
  its two redundant synchronization edges. External observation initially exceeded macOS's total DOF load limit;
  smaller initializer chunks and short clauses alone did not fix it. At mesh ba80ddcc, grouping equivalent
  compatible-state transitions into bit sets enables and observes all161 program operations successfully;
  its215 kernel records remain static IR facts. Conditional submission-state emission is still being completed.
  Build18's fused eig also passes external observation:13 program operations,7 static kernel records,1 image edge
  and no redundant edges. Compiler controls retain explicit lexical parents and actual entry writer/trip identities
  through both frontends, artifacts and signature choices; equal dispatch ranges no longer stand in for ancestry.
  A deliberate mismatched existing CPU client/signature could not exercise fail-stop: macOS denied destructive
  DTrace before client execution. No stopped client remained and no machine security configuration was changed.
  Arbitrary structured-control observation must retain all compatible observable traces and their counter/object
  bindings. Reading an earlier branch's snapshot at a later probe can read reused storage and is not evidence of
  the earlier decision; that incomplete observer path is being replaced without new runtime hooks.
  These limited observations do not close items1–6.
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
  - R4's consumption of a received operand one partition at a time has historical implementations in the torch
    scope through the lowering request's item 1 and in LiteRT through torch-capture item 3. In the replacement,
    native CPU work executes, but its received-operand acquisition still covers whole transfer ranges before
    entry. The compiler must carry the existing per-member/group footprints and page masks into the compiled
    CPU iteration, including its branches and loops. This readiness granularity remains open.

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
