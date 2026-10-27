# Tensor descriptions, graph preparation and native lowering

Recorded 2026-10-26. Inventory baseline: mesh `349738c9`, metal-microbench `424b49b8`.
This is an implementation reference. The hardware and data contract remains in
[the data model](../design/data-model.md). Outstanding execution interfaces are listed in
[the removal manifest](../design/runtime-interface-removal-2026-10-25.md).

PyTorch supplies tensor descriptions, graph capture, types and differentiation. Our compiler owns
storage, work placement, physical routes, numerical lowering and submission. FX and other compiler
passes can be extended or reused without adopting PyTorch's executor or its device choices. Upstream
Metal source generation is replaceable. The target is a general tensor library across CPU, GPU, ANE
and peer computers; a model name must not select another runtime.

## Structural corrections, 2026-10-27

The current correction sequence starts with CPU contraction. `Cpu.bind` binds a
whole-matrix operation. Row-major, column-major, padded and transposed views use
SGEMM with explicit transpose and leading-dimension operands. A transposed output
swaps the two inputs and output dimensions during preparation. Other positive
strides bind BNNS matrix descriptors and caller-owned workspace during preparation.
Execution calls that prepared bulk operation; it has no per-output BLAS loop,
application packing buffer, output scatter, or workspace allocation. BNNS owns its
internal numerical implementation; binding the original pointers does not establish
that its internal algorithm performs no packing.

This removes the `mesh_cpu_product_view` dot-product loop introduced in `ac1ea093`.
Source and compiler checks cover this edit; no numerical or performance run was
made.

The next correction replaces the host start/consume tokens with input and output
views in `MetalScheduleHost` and `MetalRecordHost`. FX binds native CPU/Core ML
entry points and retains their operands in native storage. Swift declares its ANE
input and output ranges through the same recorder. Neither caller supplies a
Python runtime callback or a separate consumer marker.

During preparation, overlapping reads and writes form dependencies. The compiler
orders independent GPU work ahead of consumers that need unfinished host work.
A GPU part publishes an event only when a host operation needs it; GPU consumers
wait on their actual host predecessors. Host operations use prepared event
notifications. They do not acquire the Python GIL, inspect layouts, or allocate
commands. Empty GPU parts are omitted from submission. Adjacent replay commands
remain batched; GPU-only spans keep compact ranges without constructing the host
dependency graph. This changes our scheduling machinery, not the internals of
Accelerate or Core ML.

The third correction passes gather destinations backward through graph preparation.
`BoundGraph` resolves final outputs in reverse collective order before compiling
numerical spans. Each producer receives its final local slice as an output operand.
Ragged, standard and coalesced FX gathers bind the same indexed WRITEs. Coalesced
gathers retain one declared group. `bind_gather_views` no longer constructs a local
`Function(...copy_...)` or returns a second numerical executor.

`Function.declare`, `declare_graph` and `compiler` accept `destinations`: one tensor
or `None` per flattened result. These operands reach the existing numerical builders
and generated kernel allocations. The native IR assigns the output layout before
lowering consumers. Aliases of a placed producer become views directly, without
an upstream view-packing or buffer-copy helper. `placement.py` and `views.rebase`
carry the relation with integer shape, stride and offset arithmetic. This calculation
depends on dimension count, not element count. All of it runs during preparation.

The small-row and grouped linear kernels now receive output row and column strides;
placement into a gathered column slice does not discard those kernels. Selection,
scan, sampling and structured-transform builders accept output operands as well.
Top-k values and indices have separate layouts. Scan partial sums have their own
layout when their accumulation width differs from the output width. Linear vector
loads also account for the original views' storage offsets.

A value supplied by the caller has no producer to relocate. Its gather input must
already be the local slice of the supplied final output. Conflicting destinations,
or aliases that cannot fit in one affine view, fail preparation. Relocating a proper
subset while retaining the rest of a larger producer requires producer-domain
splitting; this pass does not invent that split. Coded-kernel declarations now
carry their binding layouts explicitly. Their indexed stores can write strided
destinations directly; bindings tied to a fixed physical layout declare that
requirement and use the preparation described below.

The [backend layout review](../design/backend-layout-review-2026-10-27.md) records
the restrictions across coded kernels, CPU/ANE bindings and LiteRT handoffs.
Contiguity is a requirement of a particular operation's input or output view.
Preparation must carry it through producer placement and page dependencies,
alongside the gathered views supported by other operations.

`kernel.declare` carries indexed binding slots and required element strides for
both inputs and outputs. `layout.place` propagates compatible whole-value
requirements back to producers before numerical spans and gather destinations
are bound. It checks the graph's alias relations using metadata before assigning
storage. Fixed external views and conflicting requirements retain their explicit
copy edges. `layout.resolve` keeps an accepted view or assigns a
required layout before recording. An unconstrained result is allocated in its
declared layout. Where a fixed external destination conflicts with a consumer's
requirement, `layout.copies` emits bit-preserving tiled copies into the common
command records. The copies name exact source and destination ranges, so the
schedule can use their individual data edges. The copy tile size is a compiler
operand, included in the whole-function solver's parameter domain.

The shared native index generator lowers shape/stride relations to constant
arithmetic and removes contiguous-axis divisions. Coded scalar accesses, vector
groups, FX row offsets and native `MatrixView` specialization use that generator.
No runtime layout descriptor is traversed. Coded entry transforms no longer ask
their callers to pack `times` or `ffn` inputs. Fixed-layout TensorOps bindings
still state their physical requirement; they do not make it a tensor-wide rule.

Collective marks still have an opaque access interface in the shared scheduler.
They preserve their declared order and join preceding host work. Removing that
conservative boundary remains part of collective access integration. Publication,
access, feedback and backend integration follow in the correction sequence.

Validation for the placement edit: Python source parses, the Torch extension and
Swift/LiteRT rank build, and the changed Metal templates compile through the library's
pipeline compiler. The ordinary linear declaration compiles into a supplied strided
output. No numerical commands, correctness suite, hourly or performance measurement
ran. These are structural changes, not measured performance claims.

## Transformations applied to PyTorch descriptions

1. `declarations.trace` uses `make_fx` and fake tensors to capture the supplied function without executing
   its payload. A backward performed by that function enters the graph; parameter gradients are retained
   as outputs. Real gradient references are restored after tracing.
2. Ragged tensor placements describe unequal local shapes and work shares. `declarations.lower` reads
   the collective operators into common calls: participants, counts, dimensions and reduction types.
3. `BoundGraph` separates numerical spans at collective boundaries, coalesces duplicate tensor attributes,
   and removes functional `wait_tensor` nodes. Native preparation retains their data relationships.
   Gather destinations are resolved backward before numerical lowering. Producers write into their final
   local output slices; output aliases preserve the chosen physical layouts.
4. `torch_mesh.compiler` lifts captured tensor attributes into explicit operands. These include weights,
   constants and prebound destinations, so functionalization can see their storage relationships.
5. The compiler functionalizes mutations and extends the selected PyTorch decomposition table with the
   shared rules in `algebra.py`. The linear-gradient rule now serves ordinary FX, ragged DTensor and
   coded-weight declarations; it expands into existing products and reductions. Input
   mutations remain represented as writes to those inputs. Shapes and views remain available to lowering.
6. Existing rewrites recognize eligible matrix products as native linears; replace a two-part input
   concatenation with two views; fuse a following `tanh`; and group independent products up to the
   configured binding capacity. These rules have explicit dtype, rank and stride conditions.
7. `native.compile_graph` uses Inductor's numerical IR, fusion scheduler and Metal source generator.
   Matrix and linear external operations instead bind our existing numerical kernels. Unsupported
   external operations fail preparation. CPU/ANE exported graphs use Core ML conversion and the shared host-record binding described below.
8. The generated Python is a preparation program. It assigns region storage, resolves views, binds
   pipelines and constants, and constructs `_stream.Commands`. It is not called during numerical execution.
9. Numerical commands and prepared collective bindings append to the common native schedule. Execution
   submits the already prepared commands. Feedback changes operands for a subsequent compilation.

`MetalScheduleFinish` closes the local commands, then uses the existing configuration channel to
exchange the compiled send and landing list ends for every cable. The two endpoints must agree before
preparation returns. GPU counts, work shares and command counts may differ across ranks. Internal
CPU/ANE graph cuts only finish their local command part; they do not exchange metadata. Execution
does not call this compiler step. Native and FX use the same boundary.

This closes a gap observed on 2026-10-26: the laptop reached its first GPU data dependency while a
Studio was still realizing its commands. The bounded wait expired. Command completion had previously
hidden that collective failure from the Torch observer. Shared outcome observation now reports it,
and final preparation exchanges actual compiled list ends instead of assuming peers compile equally fast.

Sources: [declarations.py](../rdma/torch-mesh/torch_mesh/declarations.py),
[compiler](../rdma/torch-mesh/torch_mesh/__init__.py),
[native.py](../rdma/torch-mesh/torch_mesh/native.py),
[partition.py](../rdma/torch-mesh/torch_mesh/partition.py).

Floating `topk` binds the shared `argmax.metal` radix selector through
`selection.py`. Tensor views, the selection dimension, output order and launch
geometry are preparation operands. The existing sampling API retains its
finite-only order; tensor selection includes infinities and NaNs. Sorted
outputs use the same bitonic pair steps in threadgroup storage when the device
has room, or in prepared device storage otherwise. Those steps enter ordinary
command records; no framework selection call runs with the payload.

`cumsum` binds the same threadgroup prefix sum used by that selector. Preparation
divides the scan axis into blocks, recursively scans their totals, and binds the
addition of those prefixes to the output. Views supply their original strides.
The block geometry comes from the Metal compilation operands and device limits.
Intermediate floating sums retain accumulation precision; equal-width output
storage holds its own partial sums. All levels enter the native command records
before execution, including the expert-dispatch graph's integer slot counts.

## Shared ordered graph passes

[mesh_graph.py](../rdma/mesh_graph.py) receives operation reads and writes as value identities,
result roots, required effects and existing cut positions. `partition` returns the needed operations
and boundary operands for each span. `needed` resolves backward dependencies at each definition;
`frontiers` follows reads to their preceding producer. Reusing a value name does not make an earlier
reader depend on a later write. Duplicate cuts retain empty spans. Constants remain in the frontend's
constant table, and frontend input/result declarations emit no numerical command.

FX supplies node identities and its mutation/random-effect classification. Its former `split_module`
path is removed. `declarations.sections` materializes the common partitions as FX fragments, retaining
their tensor metadata and the original fake-tensor shape environment. Native outputs bind directly to
the next fragment's input operands. Tuple/list results and their aliases retain their declared structure;
flattening for numerical lowering happens only during preparation.

LiteRT supplies tensor indices, graph outputs and configured crossing cuts to the same `partition`
function. Its adapter serializes the resulting subgraphs and typed constants. Neither frontend's
partition adapter emits a numerical operation, copy, wait, route or new execution interface.

## Tensor views in command dependencies

FX and native command operands carry their accessed byte ranges into the shared recorder.
`MetalViewReaches` computes their union from shape, byte strides and element width during preparation.
Both frontends call this function; the former Python `views.access_spans` implementation is removed.
Transfer lists still use `views.byte_spans` to retain logical element order.
A dense transpose is one range, broadcast dimensions add no repeated range, and holes in a strided
view are not reads or writes. Empty views declare no bytes instead of the remainder of their buffer.

Each binding reach names its argument index. Multiple ranges can belong to that argument, and two
arguments can alias one buffer without losing their separate invocation offsets. The shared replay
compiler carries these declarations through both thread-count and threadgroup-count launches, including
when a prepared program is recorded into a larger program. Previously thread-count replay dropped them.
The native streamed panel's direct output rows use the same range interface instead of their enclosing
rectangle. LiteRT's contiguous external tensor views still enter through `MetalRecordViews`.

This changes dependency metadata and prepared barriers, not the numerical kernels, tensor values or
transport routes. The range lists are consumed during preparation. No range search, allocation or extra
submission runs with the payload. Generic native commands which supply only a buffer still describe its
remaining extent; those callers need their actual layouts supplied before they have FX's precision.

Native normalization, residual add, scalar scale, GELU, ordinary format products and stacked products
now supply their layouts. `MatrixView` includes column strides and grouped rows. Packed weights use
their block format, row group and pitch. Head outputs name their actual query-column slice and KV rows.
Fused epilogues keep the standalone command's declared ranges when capturing its addresses, removing
the original argument indices because those indices name different bindings in the fused command.
See [native accesses and tile publication](native-access-publication.md) for the boundaries still open.

`MatrixView` also accepts empty, repeated-read and reversed affine views. Native RMS
normalization and coded finish compile their addresses with the same index generator
used by FX. Their access records consume those same view operands. RMS reduction
storage follows the selected threadgroup size instead of a separate fixed array.
Structured-transform block width is an explicit graph-construction operand; the
chosen factor fits the device's memory and sizes the specialized shader workspace.

Validation on 2026-10-26: the native library, Torch extension and Swift/LiteRT rank built. The existing
transformer example's block prepared its complete forward/backward graph at batch 2, sequence 1,024,
width 2,048 and float16, including native schedule realization. No numerical submission was made.
That preparation exposed the missing ordinary-FX use of the linear-gradient rewrite already present
in the ragged frontend; `algebra.py` now supplies it to all three tensor declarations.

## Native and LiteRT preparation

| Frontend | Description supplied | Common destination |
|---|---|---|
| FX | Tensor graph, differentiation and placement metadata | `_stream.Commands` creates `MetalCommandDescription` records. |
| Swift native | Kernels, views, constants, launch geometry, crossings and host shares | `MetalProgram.bind` creates the same command descriptions. |
| LiteRT | Imported graph fragments, tensor layouts, partitions and compiled delegate kernels | The recording device captures native commands and crossing marks into replay. |
| CPU/ANE shares | Prepared operands, work shares and producer/consumer relationships | Host records in the same native schedule. |

[metal-recording.h](../rdma/metal-recording.h) owns the common command, access, replay and schedule interfaces.
[matrix_operations.swift](../../../metal-microbench/matrix_operations.swift) supplies native composition;
[engine_recording.swift](../../../metal-microbench/engine_recording.swift) binds invocation classes;
[litert_program.m](../../../metal-microbench/litert_program.m) imports and records LiteRT stages.

The LiteRT importer, [prepare_e2b.py](../../../metal-microbench/tools/prepare_e2b.py), already provides:

- Backward dependency slicing and removal of unused computations (`needed`).
- Subgraph extraction with explicit inputs and outputs (`cut`, `subset`, `extract`).
- Composition with remapped tensor identities (`append`).
- Tensor slicing, including quantization metadata (`slice_intervals`).
- Identity tracking when a different partition changes a value (`rekeyed`).
- Separation at communication boundaries (`staged`).
- Hoisting expressions determined by prepared inputs (`save`).
- Expression-tree matching and replacement (`lower_rotations`).

Section discovery consumes the model configuration's explicit `--semantics` operand.
`configs/litert-reference-graph.json` supplies the reference export's layer and role
patterns; the importer contains no model-name switch. LiteRT records each stage once.
The former payload warm invocation before capture is removed. The recording device
handles preparation fills and captures numerical commands without dispatching them.

Head handoffs carry configured axis order and element type into explicit row count,
width, local pitch and offset operands. The native adapter binds them through
`MeshRankRows` and the shared page WRITE compiler. A head slice can span many rows;
source and destination may have different local pitches. Plain Metal external
storage is still required until other provider layouts have declared mappings.

`engines.Cpu` binds Accelerate products and the GELU/multiply epilogue into native
commands, with all weights and scratch supplied at preparation. Its former eager
Torch operation loop is removed. `Prepared` writes linear columns into their final
strided output views where the representation agrees. Separate partial results
remain for a mathematical sum or a dtype conversion; there is no concatenation
executor. Callers can supply the final output allocation.

`coreml_graph` binds any supported compiled graph's named inputs and outputs to
shared views. Required `input_layouts` and `output_layouts` mappings describe each
binding as `strided`, `contiguous`, or an explicit element-stride tuple. Preparation
uses the same layout resolver as numerical kernels; any necessary copies become
ordinary command records before or after the host record. `export_coreml_graph`
converts an exported tensor graph before
execution, with explicit deployment, precision, compute units and artifact directory.
Both use the same native Core ML binding as the existing linear/FFN graph builder.
The binding is no longer limited to one dense two-dimensional input and output.
A host record declares its input and output views; the common schedule derives
the data dependencies before execution. Native ANE commands bind their actual normalized input directly;
a sole ANE result binds its final output directly. Multiple numerical partials still
use the ordinary sum command.

## Reusable numerical families

| Family | Template body | Instance operands |
|---|---|---|
| Views and indexing | Map logical elements into existing storage | Shape, stride, permutation, slice, page indices and masks |
| Pointwise expressions | Evaluate a scalar expression over an index domain | Expression, dtype, vector width and fused inputs |
| Reductions and scans | Accumulate and combine partial results | Axes, combine operation, tiles and hierarchy |
| Contractions | Tiled multiply/accumulate and output expression | Layouts, accumulation type, tiles, splits and epilogues |
| Streamed computations | Update partial state as tiles arrive | State representation, tile order and publication boundaries |
| Sparse and packed operations | Decode or select a tile, then feed a numerical body | Format, predicate, scales and layout |
| Communication | Map produced page ranges into consumer page ranges | Participants, paths, parallel cables, cuts and publication order |
| Specialized algorithms | Compose convolution, sorting, FFT or factorization steps | Algorithm family, shape, layout and backend capability |

The existing matrix, linear, packed-weight, attention, selection and structured-transform kernels are
numerical bodies to expose through these descriptions. For example, `matrix.product` already binds
batched strides, segmented inputs, output views, bias, scaling and publication operands. A new emitter
should instantiate these bodies and their schedules, not transcribe them into separate frontend kernels.

## Prepared numerical declarations

`argmax.draw`, `argmax.retain`, structured rotations and coded matrix operations
now declare graph nodes. Their existing Metal bodies bind into `_stream.Commands`
at preparation. Their private eager calls to the framework stream are removed.
The coded kernel declaration carries input/output roles, shapes, types, constants
and launch geometry. Output aliases within one kernel share one assigned buffer.
`Matrix.coded_input` returns its generated tables as an output operand, alongside
its transformed inputs. Coded matrix width/work metadata is retained through FX
flattening, rather than read back from a device while tracing an operation.

## Joint configuration operands

`assignment.assign` uses one complete-call objective for all configured coordinates.
Fixed/work-dependent compute fits supply the prior. A joint residual uses weighted
finite differences within each source revision; observation arrival order does not
define a difference pair. The positive ridge term bounds the fit, and normalized
coordinates keep differently sized domains comparable. The solver projects its
result once onto the supplied allocation grains. It has no measured-candidate
accept/reject gate and no later independent compute reset.

`compiler_domains` adds bounded compiler parameters to the same vector as backend
shares and path shares. `assign` returns both allocation parts and compiler operands.
`Function(configuration=...)` binds those operands during graph compilation; grouped
linears, matrix tiles and fusion capacity consume them there. Engine preparation
uses the same parameter operands. The solver does not run during numerical work.
The supplied parameter domain determines which choices it can change; a tile or
layout absent from that domain is not magically searched.

## Page-mask communication boundary

`ncclMeshPageTableCreate` resolves local allocated storage before execution.
`ncclMeshWritePagesPrepare` consumes ordered `(page, first, count)` masks. Each mask
stays in one local page; each peer uses its own table and page size. Their logical
byte totals agree. No remote pointer or cross-page extent appears in this public
declaration. The NumPy frontend exposes `Pages` and `Write_pages_prepare`.

The raw `ncclMeshWritePrepare`, `ncclMeshWriteRowsPrepare` and
`ncclMeshWriteSpansPrepare` public interfaces are removed. Local tensor adapters
are compiler-only functions in `mesh-write-compiler.h`; Python, native and Torch
call that common lowering. NumPy's public arbitrary-span method is removed.
Compiled plans retain their existing route, relay and tile-readiness machinery.
Page validation and mask lowering add no numerical-loop lookup or check.

Matching hand-tuned performance requires retaining both the numerical algorithms and their schedules.
Operator coverage alone does not establish this. Performance evidence is the canonical committed-main
application measurement: whole-function latency together with useful throughput for the same workload.
No private executor, timing mode or fallback measurement is part of this compiler work.
