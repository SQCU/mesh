# Compiler and native execution: complete composition in pseudocode

Requested by the operator on 16 November 2026: write the goal's whole data flow as
compact pseudocode within fifteen minutes. This document specifies the intended
composition. Git and execution receipts record implementation progress.

## The one boundary

Each function reads explicit structured operands and returns explicit structured
results. Its internal working structures follow its original implementation.
Every downstream operand lives in the returned results or in an explicit input.
The producer and its immediate consumer share those values through views.

`R` denotes the complete collection of source and derived results. `R += result`
means append references to the complete result in the same page-table address
space, retaining earlier results. Existing extents and layouts describe each
section. Function-level results, joint plans and emitted programs occupy distinct
output storage. Each planning round retains the earlier rounds it reads. Producers
fill declared output buffers or expose their original output
storage through the existing header/buffer APIs. A page reference and extent
locate a value; its original layout describes its contents. Tensor fields use
the existing tensor/buffer/DLPack interfaces. Archive readers use the corresponding
section layouts and relocate address bindings.

The function names below denote calls to the mechanisms in the table following
the pseudocode. Their arguments abbreviate the complete original signatures:
`R` supplies the original operands and result-buffer views by reference. `+=`
describes reference extension, with payloads remaining in their original storage.
Original argument bundles pass through `*args` and `**kwargs` where their API uses them.
Frontend, numerical compiler and target choices are explicit AOT operands.
The initial collection contains the source bodies, known callgraph and shapes,
parameters/state, placement, topology/view identity, costs/rates and target
operands. Original compiler exceptions report compilation failures; numerical
status and error values belong to the numerical results.

## Compile the functions and their joint program

```text
compile(R, frontend, numerical_compiler, target, operands, outputs):
    S = source_results(frontend, operands, outputs.source)
    R += S
    F = function_results(S, numerical_compiler, target, operands, outputs.functions)
    R += F
    J = joint_result(F, operands, outputs.joint)
    R += J
    E = emit_result(F, J, target, operands, outputs.emission)
    R += E
    signature = synchronization_signature(F, J, E, outputs.signature)
    R += signature
    return Compilation(S, F, J, E, signature)
```

Each returned `Compilation` holds views of these complete results. The stage-local
pseudocode belongs with its implementation: [source and scheduling](../../../schedule-emit/docs/pseudocode.md),
[function compilation, joint planning and emission](../../../compiler-emit/docs/pseudocode.md),
and [native loading and execution](../../../native-execution/docs/pseudocode.md).
The following paragraphs identify the existing mechanisms that those stages call.

`frontend.capture` consumes supplied source functions, parameters/state and the
known outer function-call graph. It returns the complete source program with
ordered arguments/results, shapes, layouts, aliases, mutations and effects.
The source domain is AOT A1: known shapes and an outer callgraph independent of
tensor values. Input values and indexes remain invocation operands; source
numerical algorithms retain their internal conditions and iteration rules.

`GSPMD` consumes the supplied placement/sharding operands. It returns placed
functions, per-rank sections, counts/displacements, padded extents and valid
regions. The supplied placement can itself be the result of the cited HAP/HiGHS
ratio solver or chain-on-chain row partitioning. The corresponding unequal-shard
rules cover empty trailing shards, ragged values/row boundaries, halos and causal
stripes with their original token coordinates.

`collective_lowering` returns the original ring/tree/varying-count operations,
their local-copy or in-place operations, endpoint layouts and required layout
conversion operations. `numerical_compiler` lowers every assigned body through
the original compiler and cited numerical libraries. It returns typed bodies,
all library calls, exact operands/results, accesses, workspaces, numerical error
results and source correspondence. Each original function keeps its invocation
boundary; its internal operations supply further graph vertices.

`mesh_graph_pack` consumes the complete transfer demands, topology and per-link
cost/capacity operands and returns routes. `task_graph` composes numerical work,
local copies and routed page hops through their supplied accesses and source
correspondence. Graph expansion/contraction uses the canonical graph module.
Transfer views retain both endpoint layouts and each consumer's blocking or
independent-page behavior.

`target.declare` supplies logical receive/send geometry and typed result-buffer
references before typed composition and lowering. `target.lower` returns the actual additional
operations, accesses and storage for phases, publication, status, indirect work,
and target workspaces. Its source correspondence joins them to the same graph.
`target.storage_and_actor_results` extends the same graph and costs with all
lowering-produced operations and their actual storage, effects and actor records,
including the cost producer's status. The joint planner consumes that complete
result. `target.bind_solved_operands` binds the emitted operations to the solved
offsets and frame-table views. Final emission specializes those planned operations.

## The one joint planning call

`joint_result` denotes the single canonical `mesh_plan_program` or `mesh_plan_local`
call with its complete output in separate `J` storage and its references appended
to `R`. The following expands its
algorithmic responsibilities. The canonical library owns these algorithms;
the expansion identifies their operands and retained results.

```text
joint_result(F, operands, output):
    return mesh_plan_program_or_local(F, *operands, output=output)
```

The dependence producer consumes original read/write/commuting accesses and
returns causes and source identities with its relation. Closure and chains
also supply outstanding-call bounds. Maximum cycle mean, feedforward bounds and
Lam's expansion supply the simultaneously live copies and their remapped
dependence/source correspondence. One completion-time
problem covers computation and every routed hop; its priorities feed the
topological order. Lifetimes span first definition through the final numerical
read and communication completion. Input lifetimes start at entry. Received
cyclic intervals start at their execution's start as collective-values item 14
specifies; results extend through the output boundary; state and invariants follow
their owner's lifetime. The allocator consumes these solved facts for parameters,
operands, workspaces, transport, completion records and retained compiler values.
Contiguous operands receive arena offsets; paged operands receive frame mappings.
Read and write-ahead mappings retain their distinct meanings.

## Compose and emit each target from that result

```text
emit_reference(F, J, operands, output):
    writer = original_wrapper_writer_completion(F, J, operands, output.writer)
    built = original_target_compiler_and_linker(writer, operands, output.build)
    return Emission(F, J, writer, built)

compose_native(source, operands, output):
    composition = original_typed_IR_composition(source, operands, output.composition)
    optimized = original_target_optimizer(composition, operands, output.optimized)
    return Composition(source, composition, optimized)

emit_native(F, J, operands, output):
    writer = original_target_writer(F, J, operands, output.writer)
    built = original_target_compiler_and_linker(writer, operands, output.build)
    return Emission(F, J, writer, built)
```

Both targets consume the same complete placed program, bindings, schedule and
allocation results. The reference emitter completes the already prepared
numerical result using those supplied decisions. Target-specific operands extend
that interface. Composition
uses the original numerical bodies and typed acquire/publish/copy operations;
FLUX/TileLink/CoCoNet fusion and the original optimizer produce prologues,
epilogues and fused bodies. The final writers are the original FX/C++/MSL
writers, followed by their original compilers and linkers. The result includes
callable bodies with the reference `Program(scheduled, functions)` convention,
target artifacts, argument bindings and emitted-site correspondence.

An emitted native entry issues the fixed CPU/MPS/ANE and transfer operations for
its compiled scope. Received-page phases and typed access maps supply masks;
the compiled scan/compaction operations write worklists and dispatch lengths.
Source numerical branches remain inside their numerical functions. Fixed outer
ordering and binding decisions become instructions and literals.

## Load, initialize, invoke and retire

```text
save(compilation, archive):
    return archive.write_sections(compilation)

load(archive, live_resources, output):
    A = archive.map_sections()
    L = bind_page_views_and_live_resources(A, live_resources, output)
    L.functions = load_compiled_functions(A, L)
    L.initialization = execute_compiled_initializer(A, L)
    return Loaded(A, L)

invoke(loaded, inputs, execution):
    arguments = views(loaded, inputs, execution)
    return loaded.compiled_entry(*arguments)

retire_invocation(invocation):
    return deferred_reclamation(invocation.live_bindings, invocation.completion)
```

The archive contains the source and derived sections, schedule, numerical and
transport artifacts, state and page bindings. Loading maps those same results
and binds current addresses. Initialization loads each placed parameter section,
computes its constant results once for that loaded owner, and initializes the
compiled phase/count/publication state. All those operations write assigned
pages. Ordinary invocation binds changing inputs and the execution index.

The native entry's compiled operations implement this sequence:

```text
submit(function_name, execution_range) through the island's fixed sequencer
deliver consecutive sequence numbers through pending / nextdeliver
select the compiled copy residue and phase for each execution
execute each processor/cable subsequence as its operands become available
publish each produced range after its writes using release + atomic_max(t + 1)
complete each endpoint into its success/error records
return the invocation's outputs and status after its endpoints complete
```

These lines denote compiled operations and the existing bridge's fixed protocol.
The bridge owns registered transport resources and consumes compiled endpoints
and cable order. Reposting follows the previous receive and final readers.
Independent flow sets use the specified Threefry/Fisher–Yates permutation across
all route positions. The common unroll factor supplies `t mod u`; received pages
use the specified Gray phase, range counts and monotonic publication thresholds.
Each Metal commit owns its feedback record. API-required notification handlers
write those records, which the completion consumer reads. Invocation retirement
releases its live bindings after their outstanding users complete. Compiler
results retain their separate ownership until an explicit scheduled retirement.

For the reference realization, the concrete invocation consumer is
`schedule_emit.runtime.Program(compilation.S, compilation.E.functions)`. Those
functions contain the actual transformed bindings and operations produced above.

## Reuse results when supplied operands change

```text
replan(R, F, changed_operands, target, outputs):
    J = joint_result(F, changed_operands, outputs.joint)
    R += J
    E = emit_result(F, J, target, changed_operands, outputs.emission)
    R += E
    return PlannedEmission(F, J, E)
```

The caller supplies the retained function results and new planning operands.
This call creates a new joint plan and executable while the earlier function
results, joint plans and executables retain their original storage. Changes to
routes or shard shapes call their respective producers with new output storage;
their direct consumers read those returned results. Existing solved results
supply unchanged operands by view. Parameter values remain data unless the
compiler specialized a function on those values. A new membership view completes
old-view communication with its recorded outcome and supplies the next topology
operand.

The operator's 16 November clarification supplies the storage rule: “joint plan
must ultimately be a different return in a different buffering space” and
“write-by-concatenation”. An algorithm can reuse its private scratch during its
call. Its returned results persist independently of that scratch. This storage
rule preserves the operands for later parametric optimization through ordinary
function calls and explicit output buffers.

## Observe and replace

```text
observe(R, existing_client):
    signature = R.synchronization_signature
    monitor = DTrace.specialize(signature)
    observation = existing_client.run(R, monitor)
    return observation

install(completed_standalone_modules):
    replace the old compiler and execution modules with these modules
    update their direct callers to the same complete-result interfaces
    commit the replacement and its existing-client observations
```

The signature includes queue, host, bridge and kernel synchronization with counts,
delays, source dependencies and emitted sites. Its Matrix Market, DOT and terminal
printers read the same result. The external monitor compares actual events with
that signature and records its digest and source attribution. Existing clients
exercise the program domain's numerical, alias/state, order, communication and
performance properties through the sanctioned entries. The three standalone
repositories have explicit roles: `schedule-emit` supplies the executable
function/operand reference; `compiler-emit` supplies original compiler composition;
`native-execution` supplies native loading, binding and invocation.

## Mechanisms and coverage

| Pseudocode responsibility | Existing mechanism and specification |
|---|---|
| Capture and numerical compilation | `torch.export.export`, FX, original `GraphLowering` and AOT/FX emitters; typed LiteRT/ML Drift/Tint entry; [AOT A1/A8](aot-compilation-change-request.md), [source lowering 1–3](torch-source-lowering-change-request.md), [composition](compilation-composition-change-request.md) |
| Complete numerical domain | Original Torch lowering and library calls; Accelerate LAPACK drivers and their `INFO`/rank/tolerance results; JAX QDWH/eigh/SVD/expm with their original conditions and error results; SoftFloat/FDLIBM scalar and complex semantics; [source lowering 5/6](torch-source-lowering-change-request.md), [composition's target libraries](compilation-composition-change-request.md#complete-replacement-all-in-one-commit-workgroup) |
| Placement and unequal shapes | GSPMD §§3–4; HAP §5 with HiGHS; LR-PROBE/Nicol; MPI varying-count/subarray layouts; Arrow/TensorFlow ragged layout; Striped Attention; [unequal shards 1–15](unequal-shards-change-request.md) |
| Collective and route expansion | Patarasuk–Yuan, Rabenseifner, Thakur et al.; Garg–Könemann through `mesh_graph_pack`; [AOT A3/A4](aot-compilation-change-request.md), [collective values 1–6/10](problems-of-14b9a757-change-request.md), [R1/R4/R6](plan-to-bridge-change-request.md) |
| Dependence, correspondence, bounds | Bernstein/Rinard–Diniz; ISL access intersection; canonical `rdma/mesh-graph.h` relation/expansion/closure/chains; BSL cycle mean and buffer bounds; [AOT A10](aot-compilation-change-request.md), [composition 3/6](compilation-composition-change-request.md) |
| Joint order and memory | `mesh_plan_local` / `mesh_plan_program`; `mesh_graph_completion_times` (Chen/Hall/Queyranne/HiGHS), `mesh_graph_heap_order`, lifetimes, Lam expansion, Dilworth/Fulkerson, Mukerji Fatcover/left-edge, TFLite `GreedyBySizeAssignment`; [AOT A5/A6](aot-compilation-change-request.md), [R2/R5](plan-to-bridge-change-request.md), [disjoint pages](disjoint-pages-change-request.md), [capture 2/6/7](torch-capture-change-request.md) |
| Typed target expansion and fusion | `metal_ir_lower`, original Storages/Actors records, Tint IR/Clone/optimizer/writer and original CPU compiler; TVM/Halide, FLUX/TileLink/CoCoNet; [capture 3/5/8](torch-capture-change-request.md), [composition 5/S8](compilation-composition-change-request.md) |
| Archive, load and initialize | Original section readers/writers and retained owners; AOTI/target loaders; source state slicing and constant initialization; [AOT A9](aot-compilation-change-request.md), [collective values 8](problems-of-14b9a757-change-request.md), [disjoint D.3](disjoint-pages-change-request.md) |
| Invocation, transport and completion | Fixed sequencer/pending/nextdeliver and FIFO merge; Lee–Ha; MPI persistent operations; eventcounts; Gray phases/release publication; Metal 4 commits; [AOT A7/repeated execution](aot-compilation-change-request.md), [R7/R8](plan-to-bridge-change-request.md), [resident page queue](page-queue-change-request.md), [capture 4/5/9](torch-capture-change-request.md), [integration 1/2/7](integration-finishing-change-request.md) |
| Revision and retirement | Explicit compilation dependencies, original graph traversal, source correspondence; versioned topology/ViewId and RCU; [collective values 11–14](problems-of-14b9a757-change-request.md), [disjoint D.3.6](disjoint-pages-change-request.md), [composition 6/7](compilation-composition-change-request.md) |
| Synchronization observation | BSL synchronization graph/redundancy; original source/site correspondence; DTrace; Matrix Market/DOT/Lamport diagram; [synchronization 1–6](synchronization-signature-change-request.md) |

The operator's latest boundary clarification supplies the implementation style:
an algorithm chooses its private working structures and reads/writes the explicit
serializable structures at its function boundary. The existing [functional](functional-compilation-memorandum.md),
[view](access-by-view-memorandum.md) and [lowerer-interface](lowerer-interface-memorandum.md)
memoranda apply to those boundaries. This pseudocode gives their concrete
producer-to-consumer composition.
