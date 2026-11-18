# Change request: compose compiler results through their consumers

## Requirements

### 1. Finish lowering the whole assigned domain

The frontend captures every tensor/linear-algebra program in AOT A1's domain through PyTorch export and FX (Reed et al.,
*torch.fx*, MLSys 2022; PyTorch `torch.export.export`). Operator lowerings compose into a complete target program
through full conversion (MLIR, [Dialect Conversion, Modes of Conversion](https://mlir.llvm.org/docs/DialectConversion/)).
Source-lowering items 5/6 supply the numerical algorithms and their internal conditions for CPU, GPU and ANE placement.

### 2. Compose the requests on the same program's representations

The driver composes export, placement, device assignment, schedule and target lowering as functions of explicit
values, as `schedule-emit/schedule_emit/driver.py` demonstrates. The scheduled value carries the source bodies,
placement, accesses, dependencies, storage, transfers and results of analysis to lowering, execution and archive.
Unger et al., *Unity*, OSDI 2022, §1 supplies the parallel computation graph and substitutions joining computation,
parallelization and communication.

| Request | Result produced | Consumer |
|---|---|---|
| [AOT](aot-compilation-change-request.md), A1–A10 | Captured functions, calls, placement, costs and scheduled program | Target lowering, archive and execution |
| [Disjoint pages](disjoint-pages-change-request.md), A3/C3/D.3 | Lifetimes, expanded copies, allocation and completion state | Numerical and transport bindings |
| [Page queue](page-queue-change-request.md), completed mechanism | Published page-transfer entries | Resident bridge |
| [Plan to bridge](plan-to-bridge-change-request.md), R1–R8 | Placed views and ordered transfer instructions | Bridge and target program |
| [Collective values](problems-of-14b9a757-change-request.md), 1–8/10–14 | Collective views, copies, lifetime and cyclic storage results | Placement and allocation |
| [Capture](torch-capture-change-request.md), 1–9 | Typed commands, accesses, received work and invocation results | Scheduling and native emission |
| [Source lowering](torch-source-lowering-change-request.md), 1–3/5/6 | Composable numerical bodies | Joint scheduling and target compilation |
| [Unequal shards](unequal-shards-change-request.md), 1–15 | Counts, layouts, cost functions and sharding ratios | Placement, collectives and allocation |
| [Integration](integration-finishing-change-request.md), 1–4/7 | Target order, task costs and commit observations | Native execution and existing clients |
| [Synchronization](synchronization-signature-change-request.md), 1–6 | Retained synchronization graph and emitted sites | Printers and external observation |

### Numerical lowerer boundary

The reference is the operator's demonstration at `/Users/mdot/schedule-emit`, commit `f41c64a`.

**Position.** `driver.compile_module` composes `export`, `place`, `on_devices`, `schedule` and `argument_buffers`,
and returns `Scheduled(exported, modules, devices, output_processor, channels, inputs, feedforward)`
(`schedule_emit/driver.py` 14–21, `ir.py` 57–76). Capture, placement, device assignment and scheduling are
backend-independent transformations of that value. Target lowering consumes it at `lower_inductor.lower(scheduled)`
(`lower_inductor.py` 96–103), which gives each processor's scheduled GraphModule to the upstream compiler,
`torch.compile(backend="inductor")`. A custom numerical lowerer occupies that same position. Compilation selects the
custom numerical method explicitly, and that method lowers the complete assigned body. The numerical compiler
can be factored or modified at any boundary required by the joint transformations: each extracted calculation
accepts explicit typed intermediate operands and returns its complete intermediate result to the next consumer.
A custom numerical or target implementation supplies its explicit operands and cited algorithm at the required
boundary (source-lowering items 5/6). The earlier TorchInductor selection supplies numerical implementation;
item 5 governs the joint transformations that determine the bodies and target kernels, following the operator's
[17 November clarification](../output_data/standalone-replacement-20261115/workgroup.md#compiler-library-selection-before-executable-emission).

Plain Torch supplies the numerical ablation for the same graph and schedule. A Metal-native lowerer consumes the selected
operations, dependencies, bindings and allocation to encode Metal 4 command buffers and use cooperative tensors where
the selected computation calls for them. Each candidate supplies its own kernel grouping to joint selection.
[Operator turn U156](operator-turns-2026-11-04-to-06.md) · [Metal target sources](final-lowering-sources-2026-11-06.md)

**Inputs.** In the demonstration, `adapt` reads each node's `meta["val"]` shape, stride, dtype and device and the
effectful `send` and `recv` operators (`lower_inductor.py` 66–93). Inductor's own output-stride and buffer-lifetime
contracts (`keep_output_stride`, `never_reuse_but_free_buffers`) govern the adapted values. `Exported` supplies
`graph_module`, `state`, `tokens`, `user_inputs`, `outputs` and `out_spec`, and `Scheduled.inputs` orders each
processor's placeholders. Each lowerer consumes the same complete produced program operand, carrying the retained results of AOT
[A5, A6](aot-compilation-change-request.md#a5-allocation-produces-page-bindings),
[A8–A10](aot-compilation-change-request.md#a8-the-ir-operand-carries-solved-results) and items 2/6/7 above:

- each processor's body and its source correspondence;
- shape, dtype and precision; layout and alias relations; mutation, state and effect semantics;
- ordered input and result bindings;
- the dependency, page/storage/copy, transfer and execution-order results already produced for that body.

Mesh supplies its page-allocation operands from AOT A5's retained allocation result.

**Compatible operands.** The threads-and-queues, MPS, ANE and combined CPU/MPS/ANE lowerers share the
function-and-operand contract demonstrated by `lower_inductor.lower(scheduled)` and
`runtime.Program(scheduled, functions)`. Each receives the same complete callgraph and optimization-output
operand at the corresponding boundary, with source and derived results accessible through their original owners.
Target-specific requirements extend the call with explicit additional operands. The result retains callable
bodies, their bindings and target artifacts, and the reference consumer executes those bodies with the same
scheduled argument convention. This is the operator's
[16 November clarification](../output_data/standalone-replacement-20261115/workgroup.md#compatible-operands-and-the-next-integration-work),
summarized in the [lowerer interface memorandum](lowerer-interface-memorandum.md).

The executable reference supplies the concrete meaning of function-and-operand typing here. Implementations use
its actual calling conventions and values; the operator's naming/type-discipline guidance remains in force.
Codex review follows the changed producer result through both immediate lowering consumers and uses execution
through the original reference runtime as evidence of compatibility. A shared optimization's output is the
operand of that demonstration, preserving its decisions and correspondence through the reference lowering.

The lowerer receives completed AOT results and typed result-buffer references, following the operator's
[AOT result-buffer clarification](../output_data/standalone-replacement-20261115/workgroup.md#aot-result-buffers-and-ordinary-paging-operator-clarification).
The declared AOT producers fill those buffers in the compilation's dependency order; consumers of their contents
follow those producers. Page-table mappings and byte extents locate the values, and their existing layouts supply
interpretation.

**Results.** The lowerer returns compiled bodies, their argument, result and storage bindings, and their effect and
source correspondence to the same consumers. The demonstration's `runtime.Program(scheduled, functions)`
(`runtime.py` 27–75) calls the compiled functions with prepared arguments, and `archive.save` and `archive.load`
(`archive.py` 57–116) retain the `Scheduled` graphs as one `torch.export` archive per processor. Mesh's native
execution and its A9 archive consume the compiled results. A numerical expansion returns its new internal
operations, their accesses and effects, and their workspace as typed IR with explicit source correspondence. The
ordinary owners consume that IR: dependency analysis through the graph module of item 3, and allocation and
scheduling through AOT [A5, A6](aot-compilation-change-request.md#a5-allocation-produces-page-bindings) and
[A8](aot-compilation-change-request.md#a8-the-ir-operand-carries-solved-results). Each composes the expansion's
results with the retained results (item 6), and final target emission consumes their complete retained result.

**Mechanism.** Each consumer uses the solved values it receives. Typed IR carries the bodies through the upstream
compiler's passes, which keep each algorithm's internal numerical conditions and preserve the required numerical,
state, alias and dependency semantics and the source correspondence (source-lowering item 2). Native execution
applies the compiled result to the changing inputs (item 4). The demonstration's per-processor threads, queues and
Python classes provide an executable target alongside Mesh's native execution and archive consumers.

**Executable reference target.** The compilation/emission module lowers the assigned scheduled program into
functions consumed by `schedule_emit.runtime.Program(scheduled, functions)`, through the upstream compiler's
numerical lowering. The reference's `lower_inductor.lower` demonstrates this position in the data flow.
The same complete producer result supplies
the source bodies, bindings, state, aliases, effects and compiled order to the reference and native targets.
Each target realizes those semantics through its own execution mechanism. The reference target uses its existing
threads and queues; the native target uses its compiled entry and prepared page bindings. AOT A1 defines the
program domain for both lowerings. The operator's [15 November direction and assignment](../output_data/standalone-replacement-20261115/workgroup.md)
establish actual reference-runtime execution as the interface demonstration.

Placement supplies processor assignments through AOT A2, and allocation supplies storage bindings through AOT A5.
The target adapter translates those results into its target's operations; invocation binds supplied values to the
compiled interface. Mesh's retained placement and storage results determine views, copies and transfers. The
reference adapter's `torch.device` values encode its compiler-selected targets, as clarified in the
[operator's placement clarification](../output_data/standalone-replacement-20261115/workgroup.md#placement-clarification).

### 3. Each graph function and each decision is computed in one place

The graph module exposes the cited dependency, closure, contraction, matching, ordering and synchronization
algorithms through `rdma/mesh-graph.h`, with its C implementation and Python binding. Each producer returns its
result for subsequent consumers, following AOT A5/A6/A10 and Bhattacharyya, Sriram and Lee, *Optimizing
Synchronization in Multiprocessor DSP Systems*, IEEE TSP 45(6), 1997. Newly lowered operations supply their effects
and source correspondence to that same relation.

### 4. Execute the compiled decisions with the specified runtime semantics

Compilation specializes the program with its static operands using Jones, Gomard and Sestoft, *Partial Evaluation
and Automatic Program Generation*, 1993, §4.2.2: `[[p]] [d1,d2] = [[r]] d2`. Native execution applies the residual
program to the changing inputs. Lee and Ha's self-timed strategy, as stated in Bhattacharyya, Sriram and Lee §2,
preserves compiled processor assignment and actor ordering and starts an actor when its data is available.

### 5. Compose repeated behavior into one emitted program

Target lowering composes numerical bodies and their native effect calls before optimization, through the
algorithm/schedule lowering of Ragan-Kelley et al., *Decoupling Algorithms from Schedules*, SIGGRAPH 2012,
§§5.1–5.2. The resulting native entry issues the scheduled CPU, GPU and ANE operations. The operator's Apple silicon
scope and retained-result clarification are recorded in
[the operator turns](documentary-review-2026-11-13.md#positive-mechanism-form-2026-11-14).

Function preparation and joint optimization return transformable IR through the same complete-result interface
(operator's [16 November clarification](compiler-pseudocode.md#transform-ir-then-emit-the-executable)). Tensor-parallel
selection and sharding propagation supply the numerical partitions, views and collective operations through
[AOT A2](aot-compilation-change-request.md#a2-placement-supplies-sections-and-processors) and
[unequal-shards](unequal-shards-change-request.md). Collective expansion, routing and scheduling supply the local
computation, transfers, dependencies and order through AOT A4/A6. Lam's operation/copy expansion supplies the
occurrences, lifetimes and source correspondence consumed by graph substitution and allocation
([disjoint-pages C3.3](disjoint-pages-change-request.md#c33-modulo-variable-expansion-produces-copies)).
Unity's computation-graph transformations and the cited communication-fusion mechanisms consume these complete
results together with dependent-flow/coflow decisions and return rewritten numerical and communication bodies.
These decisions enter the typed IR before target kernel selection; kernel selection consumes the resulting
bodies and explicit scheduling operands. Numerical expansion exposes its new operations, effects and workspace
to the same analysis, scheduling and allocation functions. Each required compiler boundary publishes those
typed intermediate inputs and outputs, so joint transformations can compose with the original numerical passes.
Target writers and compilers emit the executable from the transformed IR and solved bindings. Each stage retains
its predecessors through the published result views.

The compiler retains equivalent numerical and communication forms in shared IR. An IR-to-IR
transformation adds its result alongside the source. Extraction applies the caller's weighting to
the represented alternatives, and the target emits the selected form. Tate, Stepp, Tatlock and
Lerner give the mechanism of retaining alternatives through rewriting and extracting for
profitability in [*Equality Saturation: A New Approach to Optimization*, §§2–3](https://cseweb.ucsd.edu/~lerner/papers/popl09.html).

### 6. Carry compilation results through their consumers

Dependency analysis returns access causes and source identities with its graph. Synchronization reduction,
allocation, fusion and command emission consume those results through their retained references. Graph
substitutions update the source correspondence as they fuse or expand operations (Unity's parallel computation
graph; AOT A10's graph algorithms). The existing producer/result/consumer connection is the unit of integration,
as specified by the operator's retained-result instruction.

Exact bounded affine access intersection is computed through ISL's typed constraints and
`isl_basic_set_is_empty` (Verdoolaege, [Integer Set Library manual](https://libisl.sourceforge.io/manual.pdf),
Creating New Sets and Relations; Properties). Pugh's *Omega Test*, 1992, §§2.1–2.3 supplies the formulation of
dependence as integer feasibility. The resulting witnesses feed the shared dependency relation.

The operator's 17 November clarification supplies the weighting operand: “if we don't know what
'singular cost function' we're going to optimize in any of these algorithms, put a higher order
function 'weighting-function-as-input' operand into them instead”. Each compilation round retains
the transformed function graph, selected schedule and corresponding allocation. Lowering consumes
that selection. A later round can use the same earlier results with different parameters.

Each solver returns its own result. A transformation can read several of these as a tuple.
For example, the dependent-flow/coflow result describes which operations may overlap, while an
execution estimate describes duration. The former constrains legal schedules. A scheduler that
needs task lengths evaluates its supplied weighting function on the constructed calls and the
relevant solver results during compilation. A lookup table can supply that function. The same
flow result remains available to fusion and target lowering. Whole-execution measurements can
guide a later weighting choice, including a descent step where the chosen parameters admit one.

### 7. Preserve the properties while compiling consumers of solved results

Lowering specializes known kinds, bindings, addresses, order and copies into concrete target/API operations
(Jones et al., §4.2.2; Halide §§5.1–5.2). Numerical conditions retain their selected-work semantics from the cited
algorithm. Ordinary invocation supplies current inputs, indexes, invocation numbers and completion state to that
compiled program (Lee and Ha's self-timed strategy).

The compiler stores source IR and derived graphs, schedules and allocation results as compact values with retained
references, following the operator's paged-storage instruction. Header functions expose views of their serialized
sections. Derived sections can be appended beside earlier results; existing ownership and lifetime rules govern them.
The executable realizes their numerical, alias, mutation, ordering and lifetime properties through its instructions.

## S8. Analysis results become lower IR; abstract operations disappear

MLIR [One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/#what-is-one-shot-bufferize) supplies the
analysis-then-rewrite composition: use-def and alias analysis decide storage, and rewriting consumes those decisions.
Here the analysis results remain accessible as the operator's retained values while target emission specializes
operations that use them.

## Complete replacement, all in one commit workgroup

LiteRT/ML Drift's typed model operations and Tint `core::ir::Module` supply retained numerical bodies and explicit
bindings. Tint's IR transforms compose those bodies and their effects before its MSL writer prints final source
(Dawn/Tint, `src/tint/lang/core/ir` and `src/tint/lang/msl/writer`). The source-to-target pipeline retains layout,
precision, quantization, aliases and indexing maps through fusion and emission.

Target lowering fuses numerical prologues/epilogues and publication/acquisition from their shared access relations
(Chang et al., [FLUX](https://arxiv.org/abs/2406.06858), 2024;
Zheng et al., [TileLink](https://arxiv.org/abs/2503.20313), 2025;
Jangda et al., [CoCoNet](https://arxiv.org/abs/2105.05720), ASPLOS 2022).
Software binary64 arithmetic, comparison and conversion use Hauser's
[Berkeley SoftFloat 3e](https://www.jhauser.us/arithmetic/SoftFloat-3/doc/SoftFloat.html), §§4/6–8;
transcendentals use the relevant [FDLIBM](https://netlib.org/fdlibm/) implementations and dependencies.
Derived source carries those libraries' notices and rounding/special-value semantics.

Integration assignments identify the program domain and its properties, then connect every applicable source
lowering through the same placement, dependency, allocation, transport and target-emission functions. Torch-export
and typed LiteRT programs supply their numerical bodies to that composition. Existing clients supply concrete
instances for observing numerical semantics, overhead, bandwidth and useful-work throughput through
metal-microbench `docs/measurement.md` (operator's [program-domain instruction](documentary-review-2026-11-13.md#program-domain-correction-14-november);
integration item 1).

## Clarification, 2026-11-14: one compilation over retained results

The existing dependency, unrolling, allocation and scheduling algorithms operate on the parameters, operands,
calls and transfers of the whole program through the same explicit interfaces (AOT A5/A6/A10; Lam 1988;
Bhattacharyya, Sriram and Lee 1997). Fusion composes their numerical bodies through graph substitution (Unity).
The operator's [retained-result and positive-mechanism turns](documentary-review-2026-11-13.md#positive-mechanism-form-2026-11-14)
provide the assignment and its vocabulary. The table in item 2 assigns each request its producer and consumer role.

## Reusable form

- **Producer and operands:** the named function and the explicit values it takes.
- **Mechanism and source:** the algorithm, data structure or method, with author/title/section or implementation symbol.
- **Result and consumer:** the value produced and the operation that consumes it.
- **Composition:** the other requests used in those cited roles.
