# Collective client coverage

This is the caller inventory for the 2026-10-24 consolidation integrated into
the `main` working trees at `~/dox/mesh` and `~/metal-microbench`. The starting file hashes are retained in microbench's
`output_data/review-20261024/unified-collectives/baseline.json`. It records
interfaces and outstanding constraints; listing a caller does not establish
that its end-to-end route, placement, or performance has been validated.

The required common boundary is a declared operation over typed byte intervals,
including source/destination views, all admitted physical paths and relay
storage, followed by one compiled execution. Array layout, framework, and
fusion may change bindings and arithmetic. They must not select a transport
with fewer cables or different lifetime capabilities. Allocation rounding and
credit accounting must not become tensor payload padding. Physical ingress
coverage and the bound/performance gap are measured over the complete function.

## Client and call inventory

Paths in the first column are relative to mesh unless prefixed `microbench:`.
Native NCCL compatibility signatures retain their public spelling. Their
compiler must supply the common routing and lifetime representation; a caller
does not gain that capability by changing an import alone.

| Client / implementation | Executed boundary | Allocation and layout | Coverage / remaining work |
| --- | --- | --- | --- |
| `rdma/mesh.py`: `LinkMap`, `routes(lmap)`, `route(lmap,a,b)`, `members(whole,among)`, `current()`, `lease(nbytes)` | CFFI map/route discovery and `ncclMemAlloc`; this file does not execute collectives | Lease returned as a Python buffer; views retain its owner | `route` is one node path, not a physical multi-path placement description. Callers must not use it to exclude otherwise reachable relays. |
| `rdma/mesh_mpi.py`: legacy buffer methods; `Comm.Plan`, `PlanDemands` | Only declaration/query helpers connect to the compiled plan. No supported NULL-stream synchronous executor; standalone WRITE wrappers are removed. | `empty` retains bridge pages. Imperative NumPy numerical callers still require whole-function capture. | `xonotic/planner/plan.py` and `solver/worker.py:MPIWorld` remain unmigrated; changing imports or substituting imperative TorchWorld does not close the gap. |
| `python/mesh/collective.py`, `_native.py`, `__init__.py`: Program block DSL (deleted 2026-10-30: a wrapper over torch_mesh with no caller) | Movement tasks prepare ordered shared exact WRITEs on one full physical communicator. Typed reductions use native all-reduce or owner-rooted reduce over the AOT union of contiguous reference intervals. `Program.copy` uses the same prepared WRITE representation. | Contributor and recipient masks describe logical subsets; every physical rank executes crossing tasks. Views/buffers are bound once at realization. Movement binds original indexed source/final views and distinct relay leases; disconnected backings use ordered WRITEs. PARTIALS keeps complete contiguous float32 records, permits gaps between records, and rejects strided record fields. Constant initialization precedes recurrent execution. | Eleven Program tests pass after the indexed movement migration. The source removes `_bound`/`_load`/`_store` packing. The shared Torch indexed binding passes a three-rank live test; a complete live Program execution is separate validation. |
| `rdma/torch-mesh/torch_mesh/backend.py`, `_stream.mm`, `fused.py`, `partition.py` | NCCL compatibility calls and prepared exact contiguous/row/indexed WRITEs all bind the common physical fragments; complete declarations carry masks and counted-exchange matrices | Bridge leases and exact row/span descriptors; partition migration keeps DMA reads and destination consumers distinct | Backend plan metadata, generic plans/values and whole-function feedback have focused tests. The shared `SpanWrites` binding passes a three-rank live test with different endpoint strides and preserved neighboring bytes. Fused bindings preserve every active physical channel. ProcessGroup Send/Recv overrides and their native dispatch cases are removed. |
| `microbench:mesh_rank.m`: `MeshRankPlan`, `MeshRankPrepare` | Every scalar collective and row WRITE is a bound `ncclMeshCall` in `CommPlan`; preparation records through `CommEncode`. | Buffer slots retain invocation phases; typed row geometry and producer/landing operands are declared before planning. | Strict source/build validation covers this consolidation; no numerical or cluster execution of the changed interface is claimed. |
| `microbench:decode_crossings.swift`: `CrossingPoint.cross`, `StepCrossings.init` | `MeshRankGroup`, `MeshRankCrossing`, `MeshRankCross` | Existing Metal buffers wrapped with byte offsets | Shares `mesh_rank.m` transport; no separate Swift routing implementation. Configuration membership remains material. |
| `microbench:litert_program.m`: `crossing_read`, `cross`, `LiteRtProgramRun` | Typed `MeshRankCrossing` and `MeshRankCross` from LiteRT/WebGPU work | `MeshRankOperand` / `MeshRankAllocate`; shared-memory Metal buffers | Shares the same crossing executor. Imported model kernels and engine waits are data-processing concerns, not a replacement communicator. |
| `microbench:metal_native.m`, `metal_recording.m`, `metal_runtime.swift`, `engine_recording.swift` | Metal encoding/recording; crossings flow through the typed rank interface; shared `ncclMeshMetalPrelude` | Recorded resources retain bindings; engine output is an operand of later crossings | No private cable routing found. Dataflow/recorded buffer lifetime and local engine synchronization require separate scrutiny from native source-DMA retirement. |
| `microbench:goal_a.swift`: complete goal-A workload | `ncclMeshCommInitRank`; equal/V `ncclReduceScatter`; `ncclAllReduce`; equal/V `ncclAllGather` | `weights.swift:leasedBuffer` uses `ncclMemAlloc`; Metal views use offsets. Private local compute scratch is not transport storage. | These signatures now reach common native physical lowering. No new live whole-function validation of this Swift client is claimed. |
| `rdma/mesh-coreml.m`: `mesh_coreml_predict(handle,in,in_shape,in_rank,out,out_shape,out_rank)` | Core ML local prediction; **no communication implementation** | Wraps caller input; requests caller output backing. If Core ML ignores output backing, explicitly copies output and returns 0 instead of 1. | This local model behavior must be included in whole-function cost/placement. It is not an acknowledgment or a transport fallback. |
| `microbench:tools/mesh/fabric_dnn.py` | `fused.writes` or native WRITE comparison, configured physical paths and producer release times; trace observer | Explicit source/landing/relay operands | Physical routing demonstration; its successful WRITE tests alone do not establish legacy collective coverage. Exact trace spans distinguish logical payload from every hop's wire bytes. |
| `rdma/torch-mesh/torch_mesh/benchmarks.py`: `bandwidth` | Every measured collective is declared once before its repeated calls. Exchange uses a declared cyclic AlltoAllV matrix over all ranks, then the common WRITE lowering. | Shared Torch leases, original send/receive operands and retained plan | No two-rank special case, P2POp, paired issue or per-call Work.wait remains. This is a diagnostic, not a different transport implementation. |
| `microbench:tools/mesh/lstm/split.py` | `partition.attach`, distribute/split, Torch broadcast, declared gathered plan; LSTM data-function execution | Shared Torch allocator and common partition methods | `trace_observation.SpanTrace` records exact span lengths; `tick_summary` matches physical cable/unit intervals across ranks. It refuses unsupported stage grouping instead of pairing by peer ordinal. |
| `examples/chain-service.py` and `microbench:tools/mesh/torch/{tp,cp,ep}.py` | Public partition/plan/Torch collectives; complete data-function demos | Shared Torch allocation and plans | Demo arithmetic and its own timing synchronization remain. `microbench:tools/mesh/probe.py` is a transparent compatibility launcher; A/B invokes selected committed demos directly, without injected waits or allocation monkeypatches. |
| `xonotic/planner/plan.py`; `xonotic/solver/worker.py:MPIWorld`, `TorchWorld` | NumPy/Torch reductions, broadcast and counted gather; complete-function declarations through shared `controller.declare`, demand query through `ncclMeshCommPlanDemands` | Matrix multiplication writes directly into the region operand, scales in place and forwards the original gather view. MPIWorld no longer constructs a contiguous temporary; TorchWorld likewise retains the original contiguous binding. | Expert/Gram reductions and counted gather are declared for the complete solve. The declaration is suspended outside the solve so request broadcasts cannot consume its slots. There is no NumPy AlltoallV method/caller in these worktrees. |
| `rdma/mesh-diagnostics.c`, `diagnostics.py`; `microbench:tools/mesh/hourly/{quiet,retire}.py` | Versioned library diagnostic entry points | Exact immutable span metadata and bridge leases | Quiet's explicitly selected WIRE is a **per-link measurement**, not an application route selection. Hourly files only import/call library helpers. Descriptor retirement includes bridge CPU readers after abrupt client death. |

## Common configuration and remaining caller gaps

| Source and signature | Current behavior | Remaining constraint |
| --- | --- | --- |
| `rdma/mesh-collective.c`: `mesh_compile_members`, `mesh_compile`, `mesh_program_check_members`; `mesh-routed.c`: `mesh_routed_compile`; `nccl-mesh.c`: `collective_steps`, `counted_steps` | Logical collective schedules retain selected tree weights and phase structure, distinguish contributors from recipients, then use the common WRITE physical path table. Gather/ReduceScatter preserve global segment IDs including zero gaps. Counted exchanges use a complete source-major matrix; standard Alltoall/Gather/Scatter derive it. A configuration-time interval proof validates reads and required outputs of raw caller programs under declared masks. | A V exchange must be declared before execution. Its matrix is supplied explicitly or exchanged once at configuration; no runtime metadata exchange or direct-only fallback is permitted. Initialization validation preserves custom programs; it does not claim arithmetic optimality or prove DMA-source retirement. |
| `microbench:tools/mesh/programs.py`: `Fabric.__init__`, `map(among)`, `relays(a,b)`, `participants(among)` | `relays` enumerates vertices on every simple physical path, including alternatives to a direct edge. `participants` separates compute nodes from relay nodes. The node routing table remains a topology/configuration operand, not the native physical allocation. | Each executable caller must actually admit these participants and supply masks. The metadata alone cannot make an excluded process relay data. |
| `microbench:tools/mesh/rates.py` | Feeds actual executed shares and complete-call observations to `assignment.assign`. | No controller restoration or candidate acceptance path. |
| `microbench:tools/mesh/grid.py` | Uses `allocate.min_max` / `allocate.time` only to initialize shares from historical per-node measurements; subsequent feedback delegates to `rates`. Preparation installs exact proposed operands. | Full-fabric admission and installed route metadata use the integrated `mesh_rank.m` boundary above. Initial local timing fits are not a whole-function optimum. |
| `rdma/assignment.py`, `whole.py`, `routing.py`, `allocate.py` | Pure assignment math and native page-plan declarations. | Solvers return operands; compiled functions own execution. |

An allocation API name alone does not certify correctness. Original input must
be immutable until every scheduled DMA read, including unposted units, retires.
Destination overwrite requires its consumers to finish. Descriptor arrays
remain pinned through bridge CPU reads and SEND/LAND completion. Local DMA
completion does not imply a remote arithmetic consumer has finished. The
[typed lifetime model](pages-and-functions.md) remains common to all clients.

## Tests and evidence limits

`rdma/tests/test_collective_clients.py` exercises the real NumPy adapter with a
recorded CFFI boundary and compiles the production `MeshRank::issue` body
against `nccl.h`. Fifteen tests pass on 2026-10-24: exact odd-byte counts,
ragged/zero shares, in-place subrange addresses, reduction operation and root,
CPU versus Metal binding identity, independent directed WRITE bindings, and shared NumPy
allocation ownership. Prepared WRITE tests additionally cover 5,003 rows with
different rank-local pitches, original-view retention, repeated execution of
one native handle without re-preparation, plan destruction before communicator
destruction, indexed gap/repeated-source lists, reversed ndarray views, and
rejection of invalid row layouts or object dtypes without packing or value reads. Objective-C execution uses UndefinedBehaviorSanitizer.
The adapters also preserve the common controller's complete-function metadata
unchanged, and reject indivisible/mismatched Alltoall extents before a native
call instead of silently losing a tail or overrunning a receive buffer.
There is no network in this adapter test.

`test_collective_members.py` compiles the production logical compiler under
AddressSanitizer/UndefinedBehaviorSanitizer. It evaluates sparse multigraph
schedules with different input/output cohorts, poisoned relay buffers,
global zero-count gaps, ranks above bit 63 and roots outside the input cohort.
Its delayed-read regression completes one reduction before reading the reverse
DMA source and proves the original source remains immutable in both compiler
entry points. Custom-program validation tests cover partial initialization,
scratch-buffer separation, missing output intervals and malformed bounds.

`test_native_members.py` extracts the actual native realization, staging,
finalization and direct-binding functions and links the production logical and
physical compilers. It checks all six cables on a four-node sparse multigraph,
selected tree weights/phase retention, nullable inactive operands guarded by
`PROT_NONE`, contributor-count Avg, and original source pointer bindings. A
valid explicit subset program passes unchanged; an incompatible all-rank raw
program fails during configuration. These are deterministic host tests of the
real implementation, not NIC timing measurements.

`test_indexed_write.py` extracts the actual native view validation, packet
compiler, placement and public preparation bodies under address/undefined
behavior sanitizers. It tests differing per-rank span counts/orders, 5,003
spans, repeated source bytes, guarded destination gaps, registration crossings,
invalid/overflowing bounds and 300 generated maps. The actual preparation runs
with payload storage protected against CPU reads and holds one common config
lock across both metadata exchanges. `test_views.py` checks the shared
shape/stride mapper against independent logical-address enumeration.

`tests/indexed-writes-physical.py` is the three-rank live regression for the
public NumPy prepared view interface. It configures all discovered paths,
retains every original lease through communicator destruction, and checks
different endpoint maps, repeated source ranges, reversed views and neighbor
guards. Its final unaligned span is 1 GiB + 19 bytes, so it must cross an actual
registration boundary; only sparse payload sentinels are initialized and
verified. Output explicitly distinguishes those verified bytes from the full
transferred payload and reports physical counters without a throughput claim.
Four local fixture tests check its mappings, guard/value corruption detection,
sparse initialization and physical counter identity. All three cases passed on all three ranks, with traffic over all six discovered
cables. The large case verified 80 destination sentinel bytes and its guards,
not every byte of its 1 GiB + 19-byte payload. Results are retained in microbench
under `output_data/review-20261024/unified-collectives/final-main/`.

The integrated `main` local `rdma/tests` discovery passed on 2026-10-24:
**221 tests** (220 discovery tests plus the final byte-lifetime regression),
including the Torch/Program indexed bindings and zero-compute TP/CP metadata.
The retained logs are microbench's
`output_data/review-20261024/unified-collectives/final-main/mesh-final-tests.log`
and `byte-proof-final-test.log`.
The microbench caller/controller suite also passed all 21 tests.

The integrated Torch live suite passed **15 cases and 96 calls per rank** on
three ranks: ragged/zero-share gathers, reductions, affine and byte migrations,
and direct indexed `SpanWrites` with different source and destination strides.
The all-cable cases verified activity on every incident cable. These checks
assert values, original storage and neighboring bytes; they make no new
throughput claim.

Live three-rank ordinary, counted, masked and mixed-group collective checks
passed **23 cases per rank**, with traffic on all six physical cables. The
tested main working source was uncommitted. The final local/cluster results
and source identities are retained in microbench's
`output_data/review-20261024/unified-collectives/final-main/validation.json`.
These are correctness and cable-coverage results, not throughput measurements.
The original bridge images were restored and the cluster reservation released.

The native compiler separates physical arrival order from local consumption
order to avoid FIFO dependency cycles. Actual source-overwrite hazards use
local DMA retirement only where required. COPY reuse instead carries a
compiled proof that the overlapping bytes were already read, including
partial-range forwarding through relays. The exact three-rank masked Avg
regression uses six cables, 16 KiB pages and 131,101 int32 elements. Negative
cases reject shifted-byte and missing-dependency overwrites.

Run from mesh with the matching microbench source checkout:

```sh
MESH_MICROBENCH_SOURCE=/path/to/metal-microbench \
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m unittest discover \
  -s rdma/tests -p 'test_collective_clients.py' -v
```

Live complete-function tests must additionally measure achieved
latency/useful bandwidth against the admitted graph's bound. Prior exact-WRITE
and fused-allgather cluster results remain evidence for those tested paths;
they cannot stand in for the remaining callers listed here.

## Directed transfer interface audit

Native `ncclSend`/`ncclRecv`, their call kinds and endpoint-only executor are
removed, as is `ncclMeshCommRoutes`' separate per-cable weighting interface.
NumPy paired methods and Torch paired dispatch are removed. The former
bandwidth benchmark consumer uses a complete declared exchange matrix; the
Program DSL uses shared prepared indexed WRITE bindings. Explicit one-way
transfers prepare WRITE bindings and configure the complete physical path
table with `ncclMeshCommWriteRoutes`.

The source audit found no other tensor Send/Recv callers in either worktree.
`mesh_nodes.send` uploads deployment files, and Xonotic's `xonwire.send` carries
socket control traffic; neither is a mesh tensor transport. Generated CFFI
output must be regenerated from the current public header during the build,
not retained as an alternate API surface.

## NumPy fixed bindings and indexed views

The standalone NumPy WRITE preparers and `WritePlan` lifecycle are removed.
All compiled clients use `ncclMeshCallWrite` within `ncclMeshCommPlan`, carrying
local `bindings`, pure typed `views`, frames and source/landing progress records.
The shared lowerer preserves affine, reversed and indexed source selections;
physical page and registration cuts remain preparation work.

`Comm.Plan` and `PlanDemands` expose declaration/query metadata, but the legacy
imperative buffer methods are not a supported synchronous fallback. The
Xonotic NumPy callers still need their numerical producer/consumer graph lowered
into the same prepared function. Current Torch and native clients record
`ncclMeshCommEncode`; their templates retain only the complete communicator
plan serializer, with no per-WRITE plan reconstruction.
