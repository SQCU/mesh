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

## Transformations applied to PyTorch descriptions

1. `declarations.trace` uses `make_fx` and fake tensors to capture the supplied function without executing
   its payload. A backward performed by that function enters the graph; parameter gradients are retained
   as outputs. Real gradient references are restored after tracing.
2. Ragged tensor placements describe unequal local shapes and work shares. `declarations.lower` reads
   the collective operators into common calls: participants, counts, dimensions and reduction types.
3. `BoundGraph` separates numerical spans at collective boundaries, coalesces duplicate tensor attributes,
   and removes functional `wait_tensor` nodes. Native preparation retains their data relationships.
   Numerical outputs bind directly to consumers, including output aliases.
4. `torch_mesh.compiler` lifts captured tensor attributes into explicit operands. These include weights,
   constants and prebound destinations, so functionalization can see their storage relationships.
5. The compiler functionalizes mutations and uses the selected PyTorch decomposition table. Input
   mutations remain represented as writes to those inputs. Shapes and views remain available to lowering.
6. Existing rewrites recognize eligible matrix products as native linears; replace a two-part input
   concatenation with two views; fuse a following `tanh`; and group independent products up to the
   configured binding capacity. These rules have explicit dtype, rank and stride conditions.
7. `native.compile_graph` uses Inductor's numerical IR, fusion scheduler and Metal source generator.
   Matrix and linear external operations instead bind our existing numerical kernels. Unsupported
   external operations fail preparation. Generic CPU FX emission is not implemented.
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

## Shared ordered graph passes

[mesh_graph.py](../rdma/mesh_graph.py) receives operation reads and writes as value identities,
result roots, required effects and existing cut positions. `needed` performs backward dependency
selection and returns operation indices in their original order, grouped by those cuts. Duplicate
cuts retain empty spans. `frontiers` identifies each selected span's external inputs and outputs;
constants remain in the frontend's constant table.

FX supplies node identities and its mutation/random-effect classification. It uses the same selection
for dead-operation removal and numerical/collective boundaries. LiteRT supplies tensor indices, graph
outputs and configured crossing cuts; subgraph serialization and typed constant slicing remain its
frontend's work. Neither pass emits a numerical operation, copy, wait, route or new execution interface.
The frontend adapters retain the existing node objects and metadata.

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

Some section discovery still depends on tensor names such as `/mlp/` and `/q_einsum/`. That must become
semantic metadata or structural matching for general import. LiteRT also still runs each stage once to
trigger allocation and upload before recording. It is not yet a wholly declarative preparation interface.

CPU/ANE composition is not general graph placement. `engines.Cpu` currently binds Torch CPU operations;
the ANE graph builder covers linear and FFN shares. `engines.Prepared` still allocates separate share
outputs and combines them. The general compiler must expose and connect these capabilities without
retaining those restrictions as permanent backend boundaries.

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

## Remaining compiler work

1. Connect index maps, aliases, value versions, accumulation types, quantization metadata and incomplete
   reduction terms with page-range reads and writes. `BlockSpec`, partial tensors, native reaches and
   routed transfers already provide parts of this description.
2. Extend functionalization and view simplification so destinations are assigned before emission and
   views remain index maps wherever consumers can use them directly.
3. Recognize numerical patterns and apply their algebraic rules. A split product may combine partial sums;
   `tanh(a + b)` cannot be rewritten as `tanh(a) + tanh(b)`. Such decisions happen during compilation.
4. Expose existing native kernels and extend generic emission for CPU/GPU and supported ANE subgraphs.
   The shared ANE linear/FFN builder is already used by both Python and microbench.
5. Choose tiles, fusion boundaries, compute shares, backend assignments, page layouts, relay paths and
   publication order together. Capabilities and whole-function observations are compiler operands.
6. Carry partial production into transport. Recording a whole kernel's completion does not expose an
   early output tile. Bind each produced range to its writes and consumers, and assign reuse from the
   existing dependencies. No receipt protocol or runtime allocation search is needed for that assignment.

Matching hand-tuned performance requires retaining both the numerical algorithms and their schedules.
Operator coverage alone does not establish this. Performance evidence is the canonical committed-main
application measurement: whole-function latency together with useful throughput for the same workload.
No private executor, timing mode or fallback measurement is part of this compiler work.
