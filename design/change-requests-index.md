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
  Item 6 requires completing IR handoffs before helper extraction and divides that connected refactoring into
  eight implementation units. The earlier implementation carried plan identities and edges into native preparation,
  capture dependencies into replay, and typed effects through transformations. Native rows reference published
  plan calls; signature composition consumes the retained plan signature. Graph contraction, projection,
  component analysis and shader-call traversal used the existing graph module, and both shader consumers used
  the parser's retained call graph. A10's concurrency bound is separate from the horizon used to size retained
  storage. The connected source changes,
  revisions, direct builds and remaining work are recorded in that run's `integration.md` and review records;
  these do not establish completion of the full assigned domain or runtime and performance requirements.
  The subsequent structural review found active semantic transformations over emitted shader strings throughout
  the recorder, beyond the three previously listed parsers. Generated target source must become the final mesh
  emission; the producers' annotations must reach those lowerings. E2B then selected the recorder's non-self-timed
  mode, which omits replay footprints and bypasses native dependency/fence preparation. Sharing the emitter has
  not completed that synchronization-policy handoff. The performance follow-up also identifies static operation,
  dtype and layout facts interpreted by the Torch path's generic shaders, serial reduction bodies, and native
  projection facts emitted as defines then read back despite their presence in `MetalNativePlan`. Completing
  source coverage alone has not completed specialization, fusion or the required execution mapping.
  The [complete replacement workgroup](compilation-composition-change-request.md#complete-replacement-workgroup)
  specifies the connected source-built provider, typed numerical/shader interfaces, eight implementation
  responsibilities and deletion of the downstream source-rewrite path in one coordinated landing. It supersedes
  the initial 6A–6H work division, not the requirements or completed reusable work.
  Working commits metal-microbench f08d3c7b/e4d2adc5 and mesh dd2062d2/4abf824c now connect a source-built
  LiteRT/ML Drift provider to typed numerical operations and Tint IR, shared region cloning, transformations,
  CPU/GPU target emission and the native executor. The recorder's generated-source interception, parsing and
  rewrite bodies and its non-self-timed bypass are removed. All five E2B stage graphs publish through this provider;
  the actual provider, compiler, recorder and E2B caller link. The existing native diagnostic executes split-K
  projection, subgroup-matrix projection and parallel argmax; its numerical records are under the new run's
  `8_runs`. This is development evidence, not the complete replacement landing.
  Subsequent working checkpoints metal-microbench caf49366 and mesh f30bf845 carry compact access views,
  separate read/write page tables and compiled E2B input operations. The continuing work replaces compact-view
  intersection with ISL's exact integer feasibility calculation and gives kernel insertion/merging one identity
  remapping owner. These source changes still need their complete consumer and operational evidence.
  Working checkpoints metal-microbench f784364a and mesh 83c1e78e record the next connected revision. All five E2B
  stages now complete lowering and target emission in the workgroup's compile-only runs after repairs to symbol,
  epilogue-operand and value-scope handoffs. The full production dependency chain builds, including ISL.
  The existing numerical diagnostic now uses the shared native program: CPU projection, GPU split-K/matrix
  projection and argmax execute. Normalization-to-projection also executes (relative RMS error 0.000216), with the
  normalization map fused into projection. The native path exposed stale runtime effects for embedded constants;
  the uniform zero-point handoff is repaired, and the neural target's embedded weights remain under revision.
  Outstanding work includes complete frontend/target coverage, control-flow and native-program execution,
  paged native CPU/write-ahead behavior, preservation of the supplied execution index through compiled inputs,
  and displacement of unjustified per-execution host waits/submissions. Grouping must retain the selected tuned
  algorithm: a temporary no-join guard prevents its loss to generic member construction while that handoff is
  completed. Sourced actor costs, whole-group fusion choices, persisted executable/preparation results and the
  observation handoffs are still being revised.
  Working checkpoints metal-microbench 73b78edf and mesh 0932fdba preserve typed algorithm selection, control-flow
  transport releases and the initial saved-artifact consumers. Persistence is still incomplete: target binary reload,
  a shared structural identity, prepared native results and collective-plan relocation are being connected. The
  interrupted checkpoint does not link until the target save/load definitions are finished. The later continuation
  and actual build results are recorded in the same workgroup directory.
  No E2B or Torch/collective operational run or corresponding performance closure on this replacement is recorded.
- [synchronization-signature-change-request.md](synchronization-signature-change-request.md): outstanding.
  The graph schema, serialization and printers, plan/native emission, composition and external monitor are present
  in the composition implementation. The review records distinguish remaining representation and observation gaps
  from implemented source; no monitored workload has established completion of items 1–6.
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
  - R4's consumption of a received operand one partition at a time: implemented, in the torch scope by the lowering
    request's item 1 and in the LiteRT scope by the torch-capture request's item 3.

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
  A1's saved/resumed executable without repeated compilation has no complete save/load consumer in the current
  Torch/native path. Publishing the bridge's serialized plan and retaining a live compiled function do not by
  themselves provide that artifact. This handoff is assigned to the representation, Torch and executor owners.
  The composition review's remaining representation, source-domain and execution-path findings still apply.

## Superseded or closed

- [disjoint-pages-change-request.md](disjoint-pages-change-request.md): D.2, D.3.1 and D.5 are superseded by the
  plan-to-bridge request; the rest is implemented.
- [page-queue-change-request.md](page-queue-change-request.md): superseded in part; the rest is implemented.
