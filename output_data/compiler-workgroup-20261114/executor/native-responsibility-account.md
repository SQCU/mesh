# Native responsibility account (native130, read-only, DRAFT)

SAME e48, Claude Opus 5.5. Mechanical account at metal-microbench `7700acc8` (native129 source `4f592b61`).

**Source unchanged.** `git status --untracked-files=no` shows no tracked modification in metal-microbench, nor in Mesh
`rdma`, `compiler` or `design`. Only this file was written. There were no builds, imports, runs or agents. The
supervisor owns structural classification and replaces any judgment here.

**Evidence level.**
- **R** = body read.
- **G** = located by grep or signature only, body not read.

The table is a draft. Rows rest on the evidence named in them.

## Manifest

- **Full read:** the composition CR, `design/compilation-composition-change-request.md` 1–170 (R).
- **metal-microbench, `METAL_IR_SRCS`** (Makefile 107): `metal_ir.mm`, `metal_clone.mm`, `metal_transform.mm`,
  `metal_lower.mm`, `metal_native.mm`, `metal_transport.mm`, `metal_neural.mm` and `metal_archive.mm`.
- **metal-microbench, `RECORDING_SRCS`** (146): `metal_capture.m` and `metal_program.m`, with `metal_program.h` and
  `metal_submission.h`.
- **Provider:** `provider/litert_mesh.cc`/`.h`, `litert.patch` and `dawn.patch`.
- **Mesh rdma:** `mesh-graph.c` (5018), `mesh-collective.c` (3860), `mesh-dataflow.c` (351), `mesh-call.c` (39),
  `mesh-metal.m`, `mesh-metal4.m` and `mesh-coreml.m`.
- **Added by the compiler:** `rdma/mesh_code.py` (101, R).
  - **Fragment path:** there is no separate fragment C symbol or file. Greps of the metal-microbench, rdma, compiler
    and schedule-emit sources for `fragment`/`mesh_fragment` find none. `docs/provider.md` (the cited spec) was
    deleted in `31ba06ec`.
- **Inactive, malformed by intent:**
  - `metal_recording.m` (3464): not in `RECORDING_SRCS`. It is the old monolithic capture/program/native route and
    still calls `mesh_view_values`, `relation_quotient` and `relation_newest`.
  - `compiler/mesh_compiler/lower_coreml.py`: line 321 reads `lower(scheduled):` (no `def`), set by `039ad8ed`
    "Restore malformed legacy code …". No importer was found.

**Key.**
- **SAME**: a call or expansion of one implementation.
- **ALT**: a separately implemented algorithm for the same question.
- **RECOMPUTE**: re-derives an unchanged result.
- **NEW**: a new operand or projection.
- **DIAG**: diagnostic only.
- **UNRES**: unresolved.

| Behavior | Canonical implementation, retained result | Alternative / inlined / recomputed / discarded | Consumers | Whole unit / open |
|---|---|---|---|---|
| **Fragment-coded weight (`mesh.fragment`)** | Op definition and fake shape: `rdma/mesh_code.py` 8–16 (R). `Matrix`/`FFN` modules carry `_mesh_constants` (36) and call `fragment` in `product` (60). `load` (85) builds `Matrix` per safetensors entry | **Three implementations of the same decode:** (1) **upstream:** `compiler/mesh_compiler/operations.py` `_fragment_decoded` 4636 (R), the torch formulation that `decompose.py` 85–87 expands on every processor (R); (2) **legacy custom compiler:** `rdma/mesh_compile.py` `_fragment` 5934 (R), a `_Region` scalar body to `_Generic`, registered in `SOURCES` 6439 and lowered through `_Builder` into native Generic ops; (3) **native:** no fragment kernel, the Generic decode is fused by `Transform` as the dense weight prologue (`metal_native.mm` `RecognizeDense` 241, `TunedWork` 626–628 "K N decodes" (R, partial)). (1) and (2) are ALT, two hand formulations of one bit layout | Upstream decompose to Inductor; `mesh_compile` `_Builder`; `_constant_origins` 6861 reads `_mesh_constants` (R) | **UNRES:** retirement of (2) depends on the legacy `mesh_compile` route (last row) |
| **Numerical body construction** | Upstream: `metal_ir_upstream`, `Operation::Upstream`, the `ConstructKernels` upstream branch, native126–129 (R in earlier units) | **ALT, reachable:** Generic ops through `Transform` (Inline 479, grouping 1287–1360) and `Construct` (`metal_lower.mm` 1210). Producers: (a) LiteRT provider (next row); (b) `rdma/mesh_compile.py` `_Region` to `_Builder` (R at 5934 and 7420–7440 only) | `Emit`, `CompileTask`, `LowerNeural` | **UNRES:** which domains keep Generic |
| **LiteRT provider (typed source route)** | `LiteRtMeshExport` 4891 runs `Open` 4768, then `Bind` 4791, then `Publish` 4819 (R). One `Lowering` owns Generic emission (`EmitAll` 441: `metal_ir_generic` and `Region`) (R) | Two front dispatchers onto that `Lowering`: `Graph` 1149, which lowers both `md::GraphFloat32` 1186 and `md::ir::IrModel` 1196 through one `Node` 2808 (SAME per op, two source readers); `Builtin` 3163 for non-delegated TFLite nodes and composites, whose `Lower` 3945 recurses into delegate partitions (R). Shared: `Lowering::MatMul`/`Join`/`Linear`/`Permute`/`Map` and free `Unary`/`Binary`. Separate per front end: `Gather` (1485 vs 3759), plus Slice/Pad/Reduce families (G: names only) | `tools/e2b_replay.m` 1027 (G) | **UNRES:** whether the ML Drift and TFLite op families should share one op lowering. Bodies past the headers were not reviewed |
| **Mapping (`Map`)** | `Map` (transform), base map at 1293 | **RECOMPUTE:** the `Transform` local mapping vector is discarded, and `ConstructKernels` 2661 calls `Map` again (1223/2290 where construction is absent). **NEW:** fused re-maps 1305/1313; 519 is an Inline candidate (R in earlier units) | `Construct`, `Footprints` 1203, `Actor` | Retain the base mapping |
| **Operation and dispatch relation** | `Decided` (`metal_ir.mm` ≈840–895): per undecided view pair, one `mesh_graph_relation_views` solve, retained. `Operational` (2242) and `Relation` (2302) use `mesh_graph_relation_decided` with `Decide`, which answers pairs from those retained relations (R). **Resolved:** 886 is the decision cache, not a re-solve | **UNRES:** `Conflict(p, vector<View>, …)` 777 solves `mesh_graph_dependences_views` unretained. Its caller is `Construct`'s barrier placement `Before` (`metal_lower.mm` 1201, R), over per-kernel effect views split by internal space. Whether these are operation effect views already answered by `Operational` (then RECOMPUTE) or new intra-kernel views (then NEW) is undetermined. The `Transform` `Conflict(p, a, b, …)` overload uses `Hazards`, which is retained (SAME) | Order, `ReplayOrder`, `Transform`, Construct barriers | Decide 777/1201 |
| **Compiler (peer) relations** | `schedule.py` `dependence` 91 (source nodes, R) and `analyse` ≈693 (call instances J with chain causes, R), both `mesh_graph.dependences_supplied`. Different vertex sets (NEW projection) | No call of `schedule.py` `solved` 156 (`mesh_plan` then `mesh_plan_save`) was found by grep (G) | `mesh_plan_program` orders, closure | Peer to confirm `solved` liveness |
| **Physical program relation and sync** | `MetalProgramPrepare` 744–785 (R): per residue, `MetalPositionAccesses` over lag × P positions (residue copies), with plan orderings (`MetalPlannedOrder`) as supplied edges. One `relation_views` DIRECTED solve, then `relation_synchronization`, then `relation_newest`. Retained, and saved (`relation_save` 1788, G). The load path (710–740, R) loads saved relations, with "no relation solved" | **NEW by operands:** vertices are positions × iterations, and accesses are residue-specific physical copies; the logical call order is consumed as supplied edges, not re-solved. **UNRES:** an alias proof that the residue conflicts add edges beyond the call relation | Fences, waits, host/sent/start waits | Supervisor |
| **Transport protocol** | GPU/CPU in-body: `Acquiring`/`Releasing` over `Protocol` (`metal_transport.mm` 741/868), `Native`/`Acquire`/`Release`, `UpstreamProtocol` (R in earlier units) | **Host (ALT by engine):** `metal_program.m` `NativeTransfer`/`NativeRelease`/`NativeSent` (≈1118–1160, G). **ANE host program (ALT):** `rdma/mesh_metal4.py` `program` 103 (R) generates host C over SENT/RECEIVED/COPIED/COMMIT that calls `mesh_hosted` callbacks. Its only consumer is `lower_coreml.py` (malformed, inactive). rdma host side: `mesh_host_publish`/`mesh_host_wait` (`mesh-collective.c` 17/66, G) | Kernels, the native program | **UNRES:** one host protocol owner |
| **Obligations** | `Obligate` 259 (`received`/`sent`) | **DIAG:** `MetalOrder` (`metal_capture.m` 38–95, R) re-scans them for a printf | `Steps`, `Transport`, `Completions` | Delete the re-scan |
| **Readiness masks and footprints** | `Mask` uses `mesh_metal_mask` (`mesh-metal.m` 119, G); `Kernel::footprints` | Generic `Footprints` 1203 (SAME per kind). The upstream producer of footprints is missing (compiler) | Passes, the upstream helper | Compiler supplies |
| **Regions, switch, If** | `RegionForest` 954 (archived, native124); `SwitchBodies` 3215; scalar If (SAME64) | **RECOMPUTE candidate:** the `Storages` span `stable_sort` 4867 vs the retained postorder; equivalence not verified | Order, `Storages`, `Transport` | Verify |
| **Work and cost** | `MetricAnalysis`/`Actor` (2066–2098); `mesh_graph_cost_times` 5623 (G) | `Work(null)` reports 0 for upstream (≈1810, defect: unknown reported as zero). `Work(p, k, …)` 3087 has no caller. `KernelCounts` 3106 is a different question | Planning, the signature | Compiler cost and index |
| **Order and replay** | `PrepareOrder` 4913 (`orderCuts`), `ReplayOrder` 2409, `levels_view` 2347 and `heap_order_view` 2371/2441 (G) | `MetalReplayPartition` (`capture` 903, R) compares switch member and arm to add PASS levels where `orderCuts` carry no kind. One dispatch is one ICB command | Plan, ICB | **UNRES:** cut kinds |
| **Lifetimes and copies** | `Storages` 4781; `ReplayPlan.copies`; `MetalReplayPeriod` ≈222 (G) | Possible lcm recompute on fresh plans | Capture, replay | Minor |
| **Network plan** | `plan_run` (`mesh-collective.c` 2528, R): rows, grouping (`mesh_declared_group` once, or the caller's), `plan_function`, `plan_graph`, sections, `plan_place`, reserve, walk/schedule, rows, state and hops written, extents, then `plan_bind_values`/`plan_bind_table` 2127/2143 and `plan_publish` 2147 (R). Entries: `mesh_plan` 2794 (no grouping); `mesh_plan_program` 2805 (orders from the given relation's edges, CONFLICT/FLOW by member kinds, R); `mesh_plan_local` 2323 (G) | **Loads, no re-solve (R):** `mesh_plan_load` 3272 and `mesh_program_load` 3679 relocate the saved pages, rows and state (`plan_saved_page`/`plan_row_place`/`plan_state_place`), then run `plan_restore`, which reuses the same bind and publish (SAME). `plan_inputs` is a SHA-256 digest check that returns ESTALE. `plan_local_get` 3456 loads the relation and closure and runs `plan_orders_valid` (validation) | `mesh_plan_load`: `rdma/mesh_compile.py` 8218 and `tools/native_bench.m` 581 only (legacy). `mesh_program_load`: `bridge.py` 78. `mesh_local_load`: `schedule.py` 1310. `mesh_plan`/`mesh_plan_save`: `schedule.py` 160/162 and `mesh_compile.py` (G) | **UNRES:** `mesh_plan_load` and the `plan_actors` archive format serve only the legacy route |
| **Graph library** | `mesh-graph.c`. Relation: `relation_root` (sort, then edges and rows) with `view_conflicts`, `_create`/`_views`/`_decided` 1364–1399 (R). Synchronization 1624, newest 1584, quotient 1485, save/load 4212/4367, search/save/load 61/137/164, closure 257, order 376, preorder 390 (G, bodies not read) | Direct consumers (G): `relation_views` in `metal_ir.mm`, `metal_program.m` and `mesh_graph.py`; `relation_decided` in `metal_ir.mm`; `relation_quotient` in `metal_capture.m` 607; `search` in `metal_ir.mm` and `metal_native.mm` 98; `preorder` in `metal_lower.mm` 2203; `order` in `mesh-collective.c` | All above | Supervisor: one implementation, ALT not determined |
| **View projections** | `mesh_view_program` (`mesh-collective.h` 292) | `mesh_view_values` 302: live callers `schedule.py`, `mesh.py`, `mesh_compile.py`, `e2b_replay.m` and inactive `metal_recording.m` (G) | Bindings | **UNRES:** supersession |
| **ANE routes** | Native `LowerNeural` (`metal_neural.mm` 1874; `metal_lower.mm` 2825; entry `metal_ir_neural` 5598 used by `metal_capture.m` 777, G) | `lower_coreml.py` with `mesh_coreml.py`/`mesh-coreml.m` and `mesh_metal4.py`/`mesh-metal4.m`: inactive, since `lower_coreml.py` is malformed. `mesh_metal4.py` has no other consumer (G). The ANE upstream artifact consumer is missing | — | Supervisor |
| **Binary64** | `Binary64Library`, `Binary64`, `Entry` tables, one callable collection (native128/129) | SAME tables | `Source`, upstream callables | None |
| **Entry, results, completion** | `MetalNativePrepare`, `Completions` 1102, `MetalProgramEnd` ≈1876 | — | Python runtime | **Open ctypes boundary** (next section) |
| **Native archive** | `metal_archive.mm` v42 `Save`/`Load` 534/580; `search_save`/`load` 420/406 and `relation_save`/`load` 483/469 (G); `metal_ir_loaded` relocation | No load-time solve found | Loaded programs | None |
| **Legacy custom compiler route** | `rdma/mesh_compile.py` (`import mesh_code` 27; `_Region` and `_Generic` bodies; `SOURCES` 6439; `_Builder` at 7430 through the `metal_ir` C API; `mesh_plan_load` 8218 and `mesh_plan_save` 8599) (R at those lines) | ALT to the upstream compiler (`compiler/mesh_compiler` with Inductor) | `mesh_compile` callers (not traced) | **UNRES:** reachability |

## Open boundary, recorded, not fixed

Native129 ctypes are **not paired**. The compiler reports that `compiler/mesh_compiler/metal.py` still lacks
`upstream_helper` (removed upstream), and that `_emitted` reads `.types` while the native
`struct metal_ir_upstream_emission` carries `parameters` (`metal_ir.h`, native127/129). I did not read either file in
this unit; the facts are the compiler's. No import or fix was made.

## Explicit limits (not reviewed in body)

- **Provider:** `Graph::Node` beyond 2830, the `Builtin` op bodies, `Parsed` 3900, and `litert.patch`/`dawn.patch`
  beyond earlier units.
- **mesh-collective:** `plan_function`, `plan_graph`, `plan_place`, `plan_schedule` and `mesh_plan_local`.
- **mesh-dataflow:** `mesh-dataflow.c` and `mesh-call.c`.
- **mesh-graph:** the bodies of `view_conflicts`, `synchronization`, `closure`, `search`, `order`, `save`, `load` and
  `quotient`.
- **mesh-metal:** `mesh-metal.m`, `mesh-metal4.m` and `mesh-coreml.m` bodies.
- **metal_program.m:** the host transport records.
- **Compiler and frontend:** Python beyond the cited lines (`metal.py`, `runtime.py`, `native.py`, `bridge.py` and
  `mesh_compile.py` callers).
- **Not done:** an alias proof for the program relation; the `Storages` equivalence; the `Before` 1201 view
  classification.

HOLD idle. Source is unchanged and no patches are assigned.
