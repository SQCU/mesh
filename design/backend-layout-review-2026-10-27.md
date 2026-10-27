# Backend operand constraints and layout integration

Recorded: 2026-10-27T18:21:34Z. Reviewed mesh `a7d0deb7` and
metal-microbench `4fb34638`, both on `main`. This is the source review for
item 9, general backend and coded-weight layout integration. The findings
below are open implementation work, not performance results.

## The required relation

An operation names the data it reads and writes and the layout its chosen
backend needs. Contiguity belongs to that relation. An ANE operation, a Metal
operation, or a CPU SIMD operation may need a contiguous part of an operand.
Another operation may read and write gathered ranges or pages. The backend's
name alone does not settle that choice.

Compilation must carry these requirements back to producers and into the page
routes. A producer or incoming WRITE can then place values in the layout the
consumer needs. The consumer's access declaration names those same values;
their production supplies its data dependencies. A contiguous tile need not
make the whole tensor contiguous or turn every tensor dependency into a
whole-tensor join. Output requirements matter just as much as input requirements.

Keep ANE and other APIs available regardless of which demos have used them
well. Cost observations guide work assignment; they do not define hardware
support. A new specialized SIMD API must be able to state its own layout needs
through the same preparation interface.

The descriptor must distinguish a packed local address range, strided local
views, and gathered page/range views. A page list does not by itself make
disjoint local addresses contiguous for an API that takes one address range.
Compilation satisfies that requirement using the mappings the backend supports.
It first uses a suitable existing view or producer destination. Where values
must occupy another physical layout, it assigns that storage and adds the needed
copies to the graph before execution. Each copy names its source and destination
pieces. Those data edges let it overlap other work as pieces arrive, and let later
work reuse source storage once its readers finish. Callers declare requirements;
they do not write staging scripts. Execution neither chooses the layout nor
allocates its storage.

## What is already represented

- `views.byte_layout` describes affine rows and indexed byte selections;
  `_stream.ViewWrites` lowers both through the common page WRITE compiler.
- `MetalViewReaches` describes the exact union of local accesses, including
  strided, reversed and repeated read views. These are dependency descriptions,
  not a promise that every numerical kernel can address those views.
- `matrix.product`, selection, scan, structured-transform preparation and native
  `rmsNormCommand` already pass strides into their numerical commands.
- `mesh_cpu_product_bind` binds SGEMM-compatible views or BNNS strided matrix
  descriptors during preparation. It does not pack each invocation's matrix.
- `Prediction`/`mesh_coreml_bind` pass named shapes and strides to Core ML input
  arrays and output backings. `coreml_graph` also binds general compiled graphs.
  This establishes what our binding expresses; it does not establish that every
  Core ML model runs every such view on ANE without an internal conversion.
- `BoundGraph` propagates final gather slices back into producers before
  numerical compilation. Its placement representation is still affine.

The hardware manifests (`etc/compute/apple*.json`), shared `metal.def`, and
`streamed.json` describe device resources, ABI constants and kernel policy.
They do not currently describe per-operation input/output layout requirements.
The existing access records describe which bytes are used, but do not fill that
missing field on behalf of a numerical API.

## Restrictions to replace with explicit operands and lowering

Locations below refer to the reviewed revisions. A restriction's presence in
source does not establish a hardware limit. Where a restriction belongs to an
implemented instruction or encoding, its declaration and the lowering that
consumes it must agree; deleting the check alone is not a correction.

| Scope and source | Current restriction | Required integration |
|---|---|---|
| [Generic kernel declaration](../rdma/torch-mesh/torch_mesh/kernel.py), `_shape`, `declare`, `prepare` (21–53) | The schema carries output shapes/types and kernel words, but no operand layout requirements. Fake outputs are packed; preparation rejects every noncontiguous output. | Carry each binding's accepted/required layout and its chosen placement into specialization. Bind output indices and input indices from the same declaration. |
| [Coded shaders](../rdma/torch-mesh/torch_mesh/streamed.metal), `streamed_couple`, `streamed_finish`, `streamed_scale_rows`, gather/panel kernels | Stores and loads embed packed layouts. `streamed_finish` has row pitch but no independent output column/page mapping. TensorOps and vector loads have their own local layout needs. | Declare those needs per binding. Extend addressing where supported and solve any required contiguous tiles in preparation. The generic rejection cannot safely disappear before this reaches the shaders. |
| [Coded frontend](../rdma/torch-mesh/torch_mesh/streamed.py), `times`, `ffn`, `_mm_left`; [structured frontend](../rdma/torch-mesh/torch_mesh/structured.py), `Rotation`, `_backward`; [sampling frontend](../rdma/torch-mesh/torch_mesh/argmax.py), `_rows` | Callers request `.contiguous()` themselves. Structured preparation already accepts strides. Coded callers still require their packed shader inputs. | Remove caller-owned packing decisions as layout requirements reach the shared compiler. Trace-time copy declarations still describe execution work unless lowering eliminates them. |
| [Collective binding](../rdma/torch-mesh/torch_mesh/_stream.mm), `Group.contiguous` (279); [graph binding](../rdma/torch-mesh/torch_mesh/declarations.py), `BoundGraph` | Ordinary collective bindings reject noncontiguous tensors; the prepared gather/WRITE route can already carry them. | Lower logical collective views through common view/page descriptions. A contiguous physical fragment may be required by a low-level instruction without requiring a contiguous logical tensor. |
| [Placement](../rdma/torch-mesh/torch_mesh/placement.py), `alias`, `Placements`; [views](../rdma/views.py), `rebase` | One affine relocation cannot express every gather or proper subset of a producer. Non-affine aliases are rejected. | Preserve logical-to-physical range/page maps and split producer domains where required. Keep shape, element identity and destination agreement as preparation relations. |
| [Swift matrix view](../../../metal-microbench/matrix_operations.swift), `MatrixView` (219–251) | Only fp16/fp32, positive extents and positive strides; grouped affine rows are supported, arbitrary gathered maps are absent. Zero-stride read broadcasts and empty views are rejected along with invalid writes. | Separate logical views from a specific instruction's typed layout. Carry indexed reads/writes where supported; derive write-conflict rules from the actual operation. |
| [Native format products](../../../metal-microbench/bootstrap.swift), `formatProduct` (265) | Inputs are declared as packed rows; outputs have row pitch but no column/page mapping. Row classes 1/2/4/8, 32-by-64 tiles, 128 threads and 10,240 scratch bytes remain local literals. | Share explicit view and kernel geometry operands with the numerical shader and its access records. Keep a packed tile specialization available without making its layout the caller's only representation. |
| [Engine declarations](../rdma/torch-mesh/torch_mesh/engines.py), `Engine.limits`, `Cpu`, `Ane`, `neural_engine_model` | Literal type sets: CPU implementation uses fp32; specialized ANE builder uses fp16 constants. The latter also fixes a macOS15 target and has a default tile of 128. `takes` knows no layout relation. | Declare operation, type, constant-weight, deployment and layout requirements together. Preserve both these specialized implementations and the general Core ML graph binding; do not promote either implementation's domain to a universal engine limit. |
| [Swift ANE binding](../../../metal-microbench/neural_engine.swift), `NeuralEngineInvocation.bind`, `installNeuralEngine` (153–231) | Packed fp16 input/output, `HIDDEN % 4 == 0`, one common row class, and an ANE share at an end of the column interval. | Consume explicit views and operation requirements. The end-share rule is also documented in `configs/e2b-engine-programs.json`; it is a configured partition policy, not an ANE hardware fact. General page/range assignments must not inherit it. |
| [Coded slicing](../rdma/torch-mesh/torch_mesh/streamed.py), `Matrix.slice`; [native weights](../../../metal-microbench/weights.swift), view constructors and `replan`; [native coded weights](../../../metal-microbench/streamed_weights.swift), `view` | Whole 32-element tiles, permutations confined to a selected interval, no rotated column cut, some whole-owner/prefix-only views, one coded output range. Ragged FFN/head moves reject stored noncoded weights. | Keep encoded block granularity in the format declaration, and represent logical cuts, masks, basis dependencies and multiple ranges separately. Rotation can require additional input values; it does not justify pretending those dependencies do not exist. |
| [Coded capacity](../../../metal-microbench/streamed_weights.swift), `streamedPrefillScratch` (816), `streamedProgram` (1077) | A literal 12,288-column allocation becomes a rejection of larger slab/column products. | Size prepared storage from declared operands and schedule. This is an implementation ceiling, absent from the hardware manifests. |
| [Coded geometry](../rdma/torch-mesh/torch_mesh/streamed.py), `admitted`, `product_t`, `from_tensors`; Swift corresponding builders | Cooperative rows must fit one selected tile; transpose geometry chooses literal 16/64 row classes and split thresholds. Hadamard limits repeat 32/64/128, 20 and 2,560 in both languages. | Declare format/instruction geometry once, tile larger logical domains, and size workspaces from the realized operation. A selected kernel instance's domain is not the whole function's domain. |
| [Coded composition](../../../metal-microbench/streamed_weights.swift), `streamedSplit` (1047), finish builder (980); [dense execution](../../../metal-microbench/dense_execution.swift), lines 28, 108, 209, 323 | Exposed partials require one K share; finish disallows a rotated gather; some normalization, convolution, cached-attention and per-layer combinations are rejected. | Describe share axes, reduction/transform steps and their data relations through ordinary commands instead of special-case composition bans. |
| [LiteRT handoff importer](../../../metal-microbench/tools/prepare_e2b.py), `crossed_heads` (871); [native binding](../../../metal-microbench/litert_program.m), `crossing_read` (694) | Configured axis/type metadata is consumed, but a head slice is allowed only when all preceding dimensions multiply to one. The native handoff then constructs one interval. | Carry row or gathered page layouts into the existing WRITE compiler. If the provider requires packed internal operands, declare that requirement and retain its physical mapping at the handoff. |
| [LiteRT partial rows](../../../metal-microbench/tools/prepare_e2b.py), `partial_rows` (1148), head-rewrite rules (519–704); [sparse frontend](../rdma/torch-mesh/torch_mesh/sparse.py), `Closure`, `SparseLinear` | Width multiple of 128 and leading axis one; head transformations limited to specific reshape/concat/contraction patterns; sparse tiles multiple of 64 and no tail tiles. | Expose actual instruction tiles and graph-axis relations. Extend lowering for masks, tails and axis transforms rather than treating supported demo shapes as capability limits. |
| [Kernel pricing](../rdma/torch-mesh/torch_mesh/streamed.py), `Fit.node`, `choose`; [Swift pricing](../../../metal-microbench/instance_fit.swift), `candidates`, `choose` | Recorded configurations filter the candidate set. Missing measurements can leave no priced implementation. Python also requires evidence for the exact chip/core count. | Distinguish required cost inputs from hardware/format feasibility. Keep missing required inputs explicit; do not silently substitute evidence or allow the observed-shape list to define what layouts/hardware the compiler supports. |
| [FX lowerer](../rdma/torch-mesh/torch_mesh/native.py), `NativeScheduler`, extern helpers; [native model loader](../../../metal-microbench/model_file.swift), initializer | The FX numerical scheduler accepts only `mps`; CPU/ANE bindings exist separately. Native model import hardcodes model names, a 1,024 window for one form, and BF16 registrations despite the broader structure configuration. | Compose declared backend records at graph boundaries; consume model/operation configuration instead of extending an allowlist of names. No framework execution substitute. |

## Checks whose underlying relation must remain

Matching contraction dimensions, matching declared element types, fixed input
identity, disjoint writes or explicit reductions, and bounds in declared storage
describe the program. They are not guessed machine capacities. Missing metadata,
missing providers and missing backend resources must still fail preparation.

`compute.metal`, the cooperative shader's thread/scratch assertions, and scan
and selection pipeline limits already consume manifests or live device facts.
The chain's literal `TILE == 8` assertion matches its current manifest and its
explicit `simdgroup_*8x8` instructions. A different matrix instruction requires
its own declared lowering, not just changing the manifest's number.
Encoding-specific constraints such as whole compressed blocks remain real for
that encoding; broader logical cuts need an explicit masked or transformed
lowering. Plainly moving an implementation ceiling into JSON would not supply
that lowering.

The LiteRT provider currently needs plain Metal storage for the imported head
view; other storage formats need their own mapping. Core ML output-backing
identity is currently checked after prediction in both native bindings. That
check exposes an unresolved provider contract; deleting it would not establish
direct output placement. Preparation must establish whatever the provider can
guarantee, and the remaining limitation must stay explicit.

## Scope and next integration

Reviewed the owned FX declarations/lowering, tensor view and collective adapters,
coded/structured/sparse/matrix/selection/scan builders and shaders, CPU/Core ML
bindings, native command/view builders and weight/ANE callers, and the LiteRT
importer/handoff. Compared restrictions with the compute manifests, native engine
and model configuration, and the APIs those files actually call. Standalone
`tile_gemm`/`moe_matmul` shape restrictions are not the rank executor's limits;
the Makefile builds those as separate programs. This is not an audit of every
application or third-party backend implementation.

The next source change must carry per-binding layout requirements and chosen
logical-to-page mappings through declaration, placement, numerical specialization
and access publication. Reuse the existing views, page WRITE compiler and command
recorder. The interface must express contiguous local tiles and gathered ranges
on both input and output, including ANE/CPU operations, without adding a runtime
layout dispatcher. No numerical run, unit suite or hourly was launched for this
review; no rejection branch has yet been removed by it.

## Implementation following the review

The generic numerical declaration now carries fixed strides and indexed binding
roles for inputs and outputs. Its blanket rejection of strided outputs is removed.
Coded output stores and the scalar/vector input transforms use compiled affine
index maps. The shared `MetalViewIndexExpression` and binding declarations also
serve FX scan/selection row offsets and native `MatrixView` specialization.
Native `streamedFinish` now consumes a view, including column strides and grouped
rows, and publishes the exact selected output ranges. Contiguous vector groups
retain vector instructions; other views use scalar element addresses.

`layout.place` carries compatible whole-value requirements back to producers
before numerical spans and gather destinations are bound. It resolves aliases
from metadata before assigning storage. `layout.resolve` binds an accepted view directly or allocates required storage
during preparation. Required copies become tiled native command records through
`layout.copies`, with exact read/write ranges. `LAYOUT_COPY_TILE_BYTES` is a compiler
operand in the existing empirical parameter domain. This is the same recorder
used by numerical work, not a staging executor. CPU-backed command operands can
also be mapped into Metal without moving their bytes.

Core ML declarations now require a layout operand for each named input and output.
The specialized ANE builder consumes the same relation. The example configuration
in `etc/compute/coreml.json` explicitly retains strided views; a specific provider
may instead declare contiguous storage or exact strides. A required copy is
scheduled around the native host record during preparation.

The table above remains the historical review baseline. Arbitrary numerical page
maps, global producer-domain splitting, provider output-backing guarantees, and
the other listed algorithm/shape restrictions remain work. This change does not
establish global optimal layout placement or a measured performance improvement.

## Constraint propagation follow-up

Recorded: 2026-10-27T19:29:47Z. A constraint is complete only when its
declaration reaches storage placement, numerical specialization, transfer
layout and dependency records wherever those stages use it. A manifest field
with a separate literal in a lowering does not complete that work. This applies
to shapes, types, alignment, contiguous ranges, masks, resource limits, tile
geometry, ownership and value lifetimes, across every backend.

| Relation | Declaration and lowering | Result |
|---|---|---|
| Coded input storage | Matrix columns/slabs and operation rows determine prepared storage, slab offsets, shader `inputs` and `inputReaches`. Callers explicitly share one prepared input among its readers. | Removed the 12,288-column workspace and `ROWS` rejection. Unrelated large inputs no longer share a global scratch buffer. Rotation results and head partials have distinct prepared storage. |
| Coded identity order | The matrix's row or column count sizes its immutable index table. | Removed the 65,536-entry table ceiling. |
| Native affine views | `MatrixView` supplies signed strides, grouped rows, type and offset to both `MetalViewIndexDeclaration` and `MetalViewReaches`. | Zero-stride reads, reversed views and empty views are representable. RMS normalization uses this mapping; its separate unsigned stride constants are removed. Reduction workspace follows the selected threadgroup size. |
| Structured transforms | `FWHT_BLOCK_WIDTH` is a graph-construction preference. The requested row factors into blocks within live threadgroup memory. Each chosen width specializes the shader's actual array. | Removed the fixed 4,096-value shader array. An 8,192-value specialization prepared on the local device. |
| Accepted strided operands | Structured transforms and their backward declaration, sampling, and CPU/MPS shard preparation preserve views. Existing shader words and Accelerate descriptors consume their actual strides. | Removed extra `.contiguous()` calls. Required dtype/device conversion remains explicit. A non-affine sampling reshape can still require a compiler copy. |
| LiteRT head handoffs | Configuration supplies axis order and element type. The importer emits selected row count, width, pitch and offset. The native adapter passes these to `MeshRankRows`, then `mesh_compile_rows`. | Removed the leading-dimension-product-of-one restriction. Each peer's local pitch reaches the shared page WRITE compiler; gaps are not transferred. Missing element geometry fails preparation. |

Prepared storage is retained by the recorded commands. None of these builders
runs during payload execution. A value shared by several readers is an explicit
operand; equal buffer sizes do not establish that two live values may alias.
General native storage coalescing still needs the compiler's value-lifetime
relations. There is no new runtime scratch cache or allocation path.
An explicit transformed-input destination also no longer suppresses rotation
lookup-table production when the selected product consumes that table.
The shared reduction now consumes `MESH_SIMD_WIDTH` for its lane grouping,
replacing its separate shift/mask literals.

The review remains open for arbitrary numerical page maps, producer splitting,
ordinary contiguous-only collective bindings, coded block cuts and rotated
subdomains, partial-share reductions, native format-product geometry, specialized
CPU/ANE type and deployment declarations, native ANE partition assumptions,
non-plain LiteRT storage mappings, sparse tails, LiteRT partial-row transforms,
cost-evidence feasibility filters, and placement across all local backends.
The table above the follow-up retains the concrete source locations. These are
unfinished lowerings, not exceptions to the declaration requirement.

Native and extension builds, Python syntax, and Metal shader preparation passed.
RMS specializations included repeated/reversed reads and grouped output layouts;
FWHT prepared both a small block and the device's 8,192-value block. These were
compilation steps only. No numerical workload, unit suite or hourly was run.
