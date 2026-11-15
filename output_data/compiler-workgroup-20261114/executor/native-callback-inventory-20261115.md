Mesh’s required architecture represents allocation, lifetimes, placement, dependencies, and schedules as explicit data because these are solved ahead of execution and must remain available to later transformations.
Producers publish complete results in paged storage, and consumers access those same results by view, preserving their identity and page-table references.
Compilation therefore composes functions of explicit operands and represented effects, with original objects or argument bundles passed through existing interfaces using `*args` and `**args` where appropriate.
This is a stronger functional discipline than ordinary Python programming: every agent must recognize copying or reconstructing existing results as a departure from the specification and correct its work.

Every descendant function accesses produced structures through views of the same storage, with their page-table references carried intact.
Passing a view preserves access to the complete structure while each consumer dereferences exactly what its algorithm uses.
Packing, unpacking, or selecting fields into smaller replacement structs along that chain is unnecessary because the complete structure is never passed by copy.
Throughout this integration, agents use access by view to avoid spending memory bandwidth and stack work on duplicated representations, especially across Apple Silicon’s bandwidth-limited interconnect.

# Native callback inventory — 15 November 2026

SAME131 session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73, executing only `native-callback-sweep-20261115.md`. Each row is one
marker occurrence in the current source, classified from the marked source itself: a native marker directly after a lambda's
or block's opening brace is that lambda or block; one at a top-level function body's opening brace is that named function;
Swift and Python markers by their own syntax. Each row's consumer and phase are those of the nearest explanatory comment above it.

Coverage is source-only: static lexical and AST reading and editing. No native build, callback execution, import or
workload ran (see Process). The initial 645-site sweep is committed in Mesh `d9a201d2`, Metal `2c634181` and canonical
Dawn `09457b098`. The Codex provider-patch follow-through below adds 53 sites, committed in Metal `1703f8df`, with original
LiteRT `fbde6d03`, XNNPACK `877af3f` and MLDrift `239994d` source histories. The canonical LiteRT checkout now consumes
that same committed source. No Claude was launched or resumed for this follow-through.

## Markers

- Native C/C++/Objective-C: at each callback body's entry, `void (*mesh_callback_disabled)(void) = &mesh_callbacks_are_outside_spec;`,
  the address of an undeclared identifier: a compile diagnostic in C, C++ and Objective-C, with no dereference.
- Swift: `_ = mesh_callbacks_are_outside_spec;` at closure entry (after `in` where the closure has a signature); operator and
  key-path function operands are wrapped where they are passed, such as `reduce(1, mesh_callbacks_are_outside_spec(*))`. Both are
  unresolved identifiers: compile errors.
- Python (`compiler/mesh_compiler/metal.py`): the undefined callee `mesh_callbacks_are_outside_spec(...)` encloses the original
  expression at the callback's construction or registration. `_RELEASED_OBJECT = mesh_callbacks_are_outside_spec(RELEASE(_DECREF))`
  fails at module construction, before the outward callback exists or crosses the C ABI.
- Comment on the line above each marked line: `mesh callback (<phase>; <consumer>): callback inversion of control is outside
  the working specification, and runtime callback overhead is inadmissible in performance-critical code`.
- Original bodies are retained. Deleting the comment lines and markers (and unwrapping the Swift and Python wrappers)
  reproduces every marked file's committed source exactly; for metal.py, its original AST.

## Counts

698 authored markers in this native-owned inventory: the initial 645 (Metal409, Mesh native/Swift185, metal.py21, Dawn30) plus53 in the maintained LiteRT provider patch (LiteRT33, XNNPACK13, MLDrift7). These counts do not include the peer compiler/frontend inventories. The original645 included three finalizer additions after the642 check: `MetalRecorderActorsFree` in active metal_capture.m and inactive metal_recording.m, and `mesh_local_free` in rdma/mesh-collective.c.

Reachability of metal_recording.m: Makefile line 156 builds .build/libmetal_recording.dylib from `RECORDING_SRCS = metal_capture.m metal_program.m`, and no Makefile or Bazel target compiles metal_recording.m. Its 19 markers sit in a retained source that no target currently builds. They are marked because it is retained Mesh-serving source, but no current consumer reaches them; the active counterparts are in metal_capture.m and metal_program.m.

| Kind | Markers |
|---|---|
| lambda | 349 |
| named function | 184 |
| supplied-object virtual method | 6 |
| provider callback handoff | 6 |
| Swift closure | 68 |
| block | 57 |
| Python property getter | 13 |
| Swift function operand | 7 |
| Python load lambda | 3 |
| Python adoption-predicate function operand | 2 |
| Python outward ctypes callback construction | 1 |
| Python finalizer registration | 1 |
| Python adoption-predicate lambda | 1 |

| Phase | Markers |
|---|---|
| compilation/construction | 428 |
| execution | 67 |
| compilation/construction (graph analysis) | 49 |
| completion (release) | 33 |
| preparation (collective planning) | 24 |
| completion | 23 |
| preparation | 50 |
| execution (Mesh tool) | 20 |
| compilation/construction (Mesh tool) | 2 |
| preparation (completion handler) | 1 |
| compilation/construction (completion handler) | 1 |

| Repository | File | Markers | By phase |
|---|---|---|---|
| Dawn | `src/dawn/native/DawnNative.cpp` | 1 | compilation/construction 1 |
| Dawn | `src/tint/lang/core/constant/manager.cc` | 2 | compilation/construction 2 |
| Dawn | `src/tint/lang/core/intrinsic/data.cc` | 6 | compilation/construction 6 |
| Dawn | `src/tint/lang/core/ir/disassembler.cc` | 2 | compilation/construction 2 |
| Dawn | `src/tint/lang/core/ir/functional_validator.cc` | 2 | compilation/construction 2 |
| Dawn | `src/tint/lang/core/ir/structural_validator.cc` | 2 | compilation/construction 2 |
| Dawn | `src/tint/lang/msl/intrinsic/data.cc` | 2 | compilation/construction 2 |
| Dawn | `src/tint/lang/msl/writer/common/printer_support.cc` | 2 | compilation/construction 2 |
| Dawn | `src/tint/lang/msl/writer/printer/printer.cc` | 11 | compilation/construction 11 |
| Mesh | `compiler/mesh_compiler/metal.py` | 21 | compilation/construction 15; preparation 4; completion (release) 2 |
| Mesh | `examples/coreml-chain.swift` | 27 | compilation/construction 19; execution 7; completion 1 |
| Mesh | `examples/gram-chain.swift` | 31 | compilation/construction 22; execution 8; preparation 1 |
| Mesh | `examples/indexed-gather.swift` | 10 | execution 6; compilation/construction 2; preparation 1; completion 1 |
| Mesh | `examples/sync-on-remote-fill.swift` | 5 | execution 5 |
| Mesh | `rdma/mesh-collective.c` | 29 | preparation (collective planning) 24; execution 3; completion (release) 2 |
| Mesh | `rdma/mesh-coreml.m` | 1 | preparation (completion handler) 1 |
| Mesh | `rdma/mesh-events.c` | 3 | execution 3 |
| Mesh | `rdma/mesh-flow.c` | 2 | execution 2 |
| Mesh | `rdma/mesh-graph.c` | 58 | compilation/construction (graph analysis) 49; completion (release) 9 |
| Mesh | `rdma/mesh-heartbeat.c` | 1 | execution 1 |
| Mesh | `rdma/mesh-lanes.c` | 2 | execution 1; preparation 1 |
| Mesh | `rdma/mesh-link.c` | 1 | execution 1 |
| Mesh | `rdma/mesh-metal.m` | 1 | preparation 1 |
| Mesh | `rdma/mesh-metal4.m` | 4 | preparation 1; completion (release) 1; execution 1; completion 1 |
| Mesh | `rdma/mesh-ports.c` | 1 | preparation 1 |
| Mesh | `rdma/mesh-records.c` | 3 | preparation 3 |
| Mesh | `rdma/mesh-torch.mm` | 4 | completion 4 |
| Mesh | `swift/Bounds.swift` | 2 | compilation/construction 2 |
| Metal | `metal_archive.mm` | 3 | completion (release) 2; preparation 1 |
| Metal | `metal_capture.m` | 6 | preparation 3; completion 2; completion (release) 1 |
| Metal | `metal_clone.mm` | 6 | compilation/construction 6 |
| Metal | `metal_ir.mm` | 62 | compilation/construction 58; completion (release) 4 |
| Metal | `metal_lower.h` | 3 | compilation/construction 3 |
| Metal | `metal_lower.mm` | 64 | compilation/construction 64 |
| Metal | `metal_native.mm` | 49 | compilation/construction 48; preparation 1 |
| Metal | `metal_neural.mm` | 4 | execution 2; completion (release) 1; compilation/construction (completion handler) 1 |
| Metal | `metal_program.h` | 1 | completion 1 |
| Metal | `metal_program.m` | 16 | completion 6; execution 6; completion (release) 4 |
| Metal | `metal_recording.m` | 19 | completion 7; execution 5; preparation 4; completion (release) 3 |
| Metal | `metal_transform.mm` | 30 | compilation/construction 30 |
| Metal | `metal_transport.mm` | 56 | compilation/construction 56 |
| Metal | `provider/litert_mesh.cc` | 66 | compilation/construction 65; completion (release) 1 |
| Metal | `tools/e2b_replay.m` | 13 | execution (Mesh tool) 11; compilation/construction (Mesh tool) 2 |
| Metal | `tools/native_bench.m` | 7 | execution (Mesh tool) 5; completion (release) 2 |
| Metal | `tools/primitive_constants.m` | 4 | execution (Mesh tool) 4 |
| LiteRT | `ml_drift_delegate/delegate/delegate_kernel.cc` | 2 | compilation/construction 2 |
| LiteRT | `ml_drift_delegate/delegate/delegate_webgpu.cc` | 2 | execution 1; preparation 1 |
| LiteRT | `tflite/kernels/comparisons.cc` | 14 | execution 8; preparation 6 |
| LiteRT | `tflite/kernels/logical.cc` | 5 | execution 2; preparation 3 |
| LiteRT | `tflite/kernels/logical.h` | 2 | execution 2 |
| LiteRT | `tflite/kernels/mesh_selected.cc` | 2 | completion (release) 1; preparation 1 |
| LiteRT | `tflite/kernels/reshape.cc` | 3 | execution 1; preparation 2 |
| LiteRT | `tflite/kernels/select.cc` | 3 | execution 1; preparation 2 |
| MLDrift | `ml_drift/common/merge_nodes.h` | 6 | compilation/construction 6 |
| MLDrift | `ml_drift/webgpu/arguments.cc` | 1 | preparation 1 |
| XNNPACK | `src/operator-run.c` | 1 | execution 1 |
| XNNPACK | `src/subgraph/batch-matrix-multiply.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/binary.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/fully-connected.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/pack-lh.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/rope.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/softmax.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/static-constant-pad.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/static-reduce.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/static-slice.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/static-transpose.c` | 1 | preparation 1 |
| XNNPACK | `src/subgraph/unary.c` | 2 | preparation 2 |

## Marked sites

Line: the marked line in the current working tree. Symbol: a named callback body, or the function enclosing a lambda or block.

| Site | Kind | Symbol | Consumer | Phase |
|---|---|---|---|---|
| Dawn `src/dawn/native/DawnNative.cpp:60` | lambda | GetShaderModuleProgram | passed to std::shared_ptr | compilation/construction |
| Dawn `src/tint/lang/core/constant/manager.cc:203` | lambda | Manager::Zero | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/core/constant/manager.cc:205` | lambda | Manager::Zero | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/core/intrinsic/data.cc:1210` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/core/intrinsic/data.cc:1217` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/core/intrinsic/data.cc:1226` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/core/intrinsic/data.cc:1233` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/core/intrinsic/data.cc:1607` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/core/intrinsic/data.cc:1617` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/core/ir/disassembler.cc:452` | lambda | Disassembler::ValueToStyledText | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/core/ir/disassembler.cc:456` | lambda | Disassembler::ValueToStyledText | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/core/ir/functional_validator.cc:1482` | lambda | Functional::CheckConvert | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/core/ir/functional_validator.cc:1484` | lambda | Functional::CheckConvert | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/core/ir/structural_validator.cc:678` | lambda | Structural::CheckType | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/core/ir/structural_validator.cc:680` | lambda | Structural::CheckType | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/intrinsic/data.cc:192` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/msl/intrinsic/data.cc:199` | lambda | (file scope) | stored as an intrinsic table TypeMatcher match or print entry | compilation/construction |
| Dawn `src/tint/lang/msl/writer/common/printer_support.cc:153` | lambda | MslPackedTypeSizeAndAlign | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/common/printer_support.cc:157` | lambda | MslPackedTypeSizeAndAlign | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:650` | lambda | EmitBlockInstructions | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:652` | lambda | EmitBlockInstructions | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:654` | lambda | EmitBlockInstructions | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:656` | lambda | EmitBlockInstructions | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:689` | lambda | EmitValue | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:1695` | lambda | EmitType | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:1701` | lambda | EmitType | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:2134` | lambda | EmitConstant | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:2144` | lambda | EmitConstant | passed to tint::Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:2211` | lambda | EmitZeroValue | passed to Switch | compilation/construction |
| Dawn `src/tint/lang/msl/writer/printer/printer.cc:2216` | lambda | EmitZeroValue | passed to Switch | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:118` | Python outward ctypes callback construction | - | an outward ctypes RELEASE callback constructed around _DECREF, passed to each borrower | completion (release) |
| Mesh `compiler/mesh_compiler/metal.py:140` | Python load lambda | - | a load lambda and the bool adoption predicate passed to _lent | preparation |
| Mesh `compiler/mesh_compiler/metal.py:141` | Python adoption-predicate function operand | - | a load lambda and the bool adoption predicate passed to _lent | preparation |
| Mesh `compiler/mesh_compiler/metal.py:155` | Python load lambda | - | a load lambda and the bool adoption predicate passed to _lent | preparation |
| Mesh `compiler/mesh_compiler/metal.py:155` | Python adoption-predicate function operand | - | a load lambda and the bool adoption predicate passed to _lent | preparation |
| Mesh `compiler/mesh_compiler/metal.py:541` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:543` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:545` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:547` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:549` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:551` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:553` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:568` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:570` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:572` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:575` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:577` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:580` | Python property getter | - | getter registered with property | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:909` | Python finalizer registration | - | native metal_ir_binary64_free registered with weakref.finalize | completion (release) |
| Mesh `compiler/mesh_compiler/metal.py:1049` | Python load lambda | - | a load lambda and an adoption predicate lambda passed to _lent | compilation/construction |
| Mesh `compiler/mesh_compiler/metal.py:1050` | Python adoption-predicate lambda | - | a load lambda and an adoption predicate lambda passed to _lent | compilation/construction |
| Mesh `examples/coreml-chain.swift:11` | Swift function operand | - | operator * passed to reduce | compilation/construction |
| Mesh `examples/coreml-chain.swift:15` | Swift closure | - | operator * passed to reduce | compilation/construction |
| Mesh `examples/coreml-chain.swift:15` | Swift function operand | - | operator * passed to reduce | compilation/construction |
| Mesh `examples/coreml-chain.swift:59` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:62` | Swift closure | - | returned as a feature input body | execution |
| Mesh `examples/coreml-chain.swift:79` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:84` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/coreml-chain.swift:90` | Swift closure | - | registered as a TensorFunction view | execution |
| Mesh `examples/coreml-chain.swift:92` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/coreml-chain.swift:94` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/coreml-chain.swift:100` | Swift closure | - | passed to allSatisfy | compilation/construction |
| Mesh `examples/coreml-chain.swift:104` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:115` | Swift function operand | - | key path \.owner passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:117` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:123` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:125` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:127` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:131` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:133` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:137` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:139` | Swift function operand | - | key path \.operand passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:142` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:142` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:147` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/coreml-chain.swift:151` | Swift closure | - | passed to DispatchQueue.main.async | completion |
| Mesh `examples/coreml-chain.swift:161` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/coreml-chain.swift:180` | Swift closure | - | passed to withExtendedLifetime | execution |
| Mesh `examples/gram-chain.swift:18` | Swift function operand | - | key path \.bytes passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:27` | Swift function operand | - | key path \.owner passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:35` | Swift closure | - | registered as a TensorFunction view | execution |
| Mesh `examples/gram-chain.swift:37` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/gram-chain.swift:39` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/gram-chain.swift:55` | Swift closure | - | passed to allSatisfy | compilation/construction |
| Mesh `examples/gram-chain.swift:55` | Swift closure | - | passed to allSatisfy | compilation/construction |
| Mesh `examples/gram-chain.swift:57` | Swift closure | - | passed to allSatisfy | compilation/construction |
| Mesh `examples/gram-chain.swift:59` | Swift closure | - | passed to allSatisfy | compilation/construction |
| Mesh `examples/gram-chain.swift:61` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:66` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:68` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:68` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:70` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:72` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:74` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:78` | Swift closure | - | registered as a TensorFunction view | execution |
| Mesh `examples/gram-chain.swift:80` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/gram-chain.swift:82` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/gram-chain.swift:87` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:89` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/gram-chain.swift:100` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:102` | Swift function operand | - | operator + passed to reduce | compilation/construction |
| Mesh `examples/gram-chain.swift:104` | Swift closure | - | passed to mesh.constant | preparation |
| Mesh `examples/gram-chain.swift:114` | Swift closure | - | passed to contract | compilation/construction |
| Mesh `examples/gram-chain.swift:119` | Swift closure | - | passed to contract | compilation/construction |
| Mesh `examples/gram-chain.swift:124` | Swift closure | - | passed to contract | compilation/construction |
| Mesh `examples/gram-chain.swift:129` | Swift closure | - | passed to contract | compilation/construction |
| Mesh `examples/gram-chain.swift:134` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:142` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/gram-chain.swift:163` | Swift closure | - | passed to withExtendedLifetime | execution |
| Mesh `examples/indexed-gather.swift:18` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/indexed-gather.swift:25` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/indexed-gather.swift:34` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/indexed-gather.swift:47` | Swift closure | - | passed to map | compilation/construction |
| Mesh `examples/indexed-gather.swift:49` | Swift closure | - | passed to mesh.constant | preparation |
| Mesh `examples/indexed-gather.swift:56` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/indexed-gather.swift:72` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/indexed-gather.swift:88` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/indexed-gather.swift:91` | Swift closure | - | passed to DispatchQueue.main.async | completion |
| Mesh `examples/indexed-gather.swift:102` | Swift closure | - | passed to withExtendedLifetime | execution |
| Mesh `examples/sync-on-remote-fill.swift:17` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/sync-on-remote-fill.swift:24` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/sync-on-remote-fill.swift:30` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/sync-on-remote-fill.swift:39` | Swift closure | - | registered as a TensorFunction CPU body | execution |
| Mesh `examples/sync-on-remote-fill.swift:47` | Swift closure | - | passed to withExtendedLifetime | execution |
| Mesh `rdma/mesh-collective.c:18` | named function | mesh_host_publish | stored in the program run hooks (metal_program.m, metal_recording.m) | execution |
| Mesh `rdma/mesh-collective.c:30` | named function | mesh_host_arrived | stored in the program run hooks (metal_program.m, metal_recording.m) | execution |
| Mesh `rdma/mesh-collective.c:55` | named function | mesh_host_sent | stored in the program run hooks (metal_program.m, metal_recording.m) | execution |
| Mesh `rdma/mesh-collective.c:186` | named function | plan_key_order | registered with qsort | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:194` | named function | plan_end | registered with mesh_graph_reach | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:199` | named function | plan_leaf | registered with mesh_graph_flow | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:203` | named function | plan_at | registered with mesh_graph_order and mesh_graph_rows | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:206` | named function | plan_tree_parent | registered with mesh_graph_rows | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:211` | named function | plan_tree_up | registered with mesh_graph_climb | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:251` | named function | plan_edge_tail | registered with mesh_graph_rows | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:352` | named function | plan_head | registered with mesh_graph_lags | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:354` | named function | plan_lag | registered with mesh_graph_lags | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:505` | named function | plan_alone | registered with mesh_graph_chains | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:661` | named function | plan_intersect | registered with mesh_graph_greedy_by_size | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:918` | named function | plan_path_step | registered with mesh_graph_climb | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:936` | named function | plan_entry_order | registered with qsort | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1129` | named function | plan_room_head | registered with mesh_graph_lags | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1134` | named function | plan_room_lag | registered with mesh_graph_lags | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1190` | named function | plan_tell_key | registered with mesh_graph_rows | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1598` | named function | plan_cut | registered with mesh_strided_extents | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1609` | named function | plan_start | registered with mesh_strided_extents | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1616` | named function | plan_cut_order | registered with qsort | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1695` | named function | plan_touch_order | registered with qsort | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1701` | named function | plan_member_order | registered with qsort | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:1710` | named function | plan_cover_order | registered with qsort | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:2097` | named function | plan_together | registered with mesh_graph_greedy_by_size | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:2203` | named function | plan_bytes_release | registered with mesh_graph_relation_load and mesh_graph_archive_read | completion (release) |
| Mesh `rdma/mesh-collective.c:2244` | named function | plan_entry_demand | registered with mesh_graph_rows | preparation (collective planning) |
| Mesh `rdma/mesh-collective.c:2501` | named function | mesh_local_free | registered with weakref.finalize by compiler/mesh_compiler/schedule.py | completion (release) |
| Mesh `rdma/mesh-coreml.m:99` | block | mesh_coreml_plan | passed to completionHandler: | preparation (completion handler) |
| Mesh `rdma/mesh-events.c:121` | named function | peer_terminated | registered with IOServiceAddMatchingNotification | execution |
| Mesh `rdma/mesh-events.c:143` | named function | peer_arrived | registered with IOServiceAddMatchingNotification | execution |
| Mesh `rdma/mesh-events.c:195` | named function | fault_heard | registered with sigaction | execution |
| Mesh `rdma/mesh-flow.c:2024` | named function | port_progress | registered with pthread_create | execution |
| Mesh `rdma/mesh-flow.c:2290` | named function | plan_remove_at | registered with mesh_graph_row_visit | execution |
| Mesh `rdma/mesh-graph.c:26` | named function | mesh_graph_row_head | registered with mesh_graph_breadth, mesh_graph_search and mesh_graph_delayed_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:29` | named function | mesh_graph_relation_head | registered with mesh_graph_search | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:40` | named function | mesh_graph_search_free | registered by rdma/mesh_graph.py Owned weakref.finalize and as a std::shared_ptr deleter | completion (release) |
| Mesh `rdma/mesh-graph.c:207` | named function | mesh_graph_closure_free | registered by rdma/mesh_graph.py Owned weakref.finalize | completion (release) |
| Mesh `rdma/mesh-graph.c:256` | named function | closure_descending | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:605` | named function | depth_head | registered with mesh_graph_order and mesh_graph_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:678` | named function | flow_same | registered with mesh_graph_keyed_find | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:754` | named function | mesh_packed_heavier | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:762` | named function | pack_same | registered with mesh_graph_keyed_find | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1034` | named function | lags_before | registered with heap_up and heap_down | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1101` | named function | contract_tail | registered with mesh_graph_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1105` | named function | contract_head | registered with lags_walks | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1111` | named function | contract_lag | registered with lags_walks | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1190` | named function | quotient_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1215` | named function | quotient_tail | registered with quotient_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1217` | named function | quotient_head | registered with quotient_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1299` | named function | mesh_graph_relation_free | registered as a std::shared_ptr deleter | completion (release) |
| Mesh `rdma/mesh-graph.c:1364` | named function | relation_key_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1374` | named function | witness_key_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1547` | named function | relation_tail | registered with mesh_graph_rows, quotient_solve and synchronization_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1551` | named function | relation_head | registered with mesh_graph_rows, quotient_solve and synchronization_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1854` | named function | closure_before | registered with mesh_graph_chains | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:1888` | named function | cover_key_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2073` | named function | completion_key_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2079` | named function | completion_edge_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2298` | named function | order_before | registered with heap_up and heap_down | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2344` | named function | flat_degree | stored in a mesh_graph_rows_view | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2349` | named function | flat_head | stored in a mesh_graph_rows_view | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2464` | named function | zero_row | registered with mesh_graph_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:2708` | named function | mesh_graph_context_free | registered by rdma/mesh_graph.py Owned weakref.finalize and as a std::shared_ptr deleter | completion (release) |
| Mesh `rdma/mesh-graph.c:2752` | named function | mesh_graph_domain_free | registered as a std::shared_ptr deleter | completion (release) |
| Mesh `rdma/mesh-graph.c:3016` | named function | mesh_graph_access_map_free | registered as a std::shared_ptr deleter | completion (release) |
| Mesh `rdma/mesh-graph.c:3099` | named function | access_point | registered with isl_set_foreach_point | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3114` | named function | access_ascending | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3251` | named function | view_hull_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3257` | named function | view_edge_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3275` | named function | view_decided_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3551` | named function | redundant_tail | registered with redundant_solve and synchronization_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3553` | named function | redundant_head | registered with redundant_solve and synchronization_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3555` | named function | redundant_delay | registered with redundant_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3569` | named function | synchronization_delay | registered with redundant_solve | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3660` | named function | signature_key_before | registered with qsort and graph_lower | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3668` | named function | signature_dependence_before | registered with qsort and graph_lower | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3676` | named function | signature_task_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3698` | named function | signature_tail | registered with mesh_graph_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3700` | named function | signature_head | registered with mesh_graph_lags | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3705` | named function | signature_lag | registered with mesh_graph_lags | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3711` | named function | signature_processor | registered with mesh_graph_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3716` | named function | signature_image_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:3738` | named function | archived_released | stored as a release member | completion (release) |
| Mesh `rdma/mesh-graph.c:3880` | named function | scopes_released | stored as a release member | completion (release) |
| Mesh `rdma/mesh-graph.c:3889` | named function | scope_key_before | registered with qsort and graph_lower | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:4294` | named function | proof_before | registered with qsort and graph_lower | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:4914` | named function | relation_root_release | stored as a release member | completion (release) |
| Mesh `rdma/mesh-graph.c:5101` | named function | compose_key_before | registered with qsort and graph_lower | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:5584` | named function | intersect_before | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:5611` | named function | greedy_compare_by_size | registered with qsort | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-graph.c:5702` | named function | mesh_graph_relation_tail | registered with mesh_graph_delayed_rows | compilation/construction (graph analysis) |
| Mesh `rdma/mesh-heartbeat.c:324` | named function | noted | registered with sigaction | execution |
| Mesh `rdma/mesh-lanes.c:81` | block | link_job | passed to dispatch_async | execution |
| Mesh `rdma/mesh-lanes.c:110` | block | device_open | passed to dispatch_async | preparation |
| Mesh `rdma/mesh-link.c:110` | named function | control_watch | registered with pthread_create | execution |
| Mesh `rdma/mesh-metal.m:119` | named function | mask_same | registered with mesh_graph_keyed_find | preparation |
| Mesh `rdma/mesh-metal4.m:48` | block | begin | passed to dispatch_once | preparation |
| Mesh `rdma/mesh-metal4.m:107` | named function | mesh_metal4_release | registered by rdma/mesh_metal4.py | completion (release) |
| Mesh `rdma/mesh-metal4.m:146` | block | mesh_metal4_after | passed to dispatch_sync | execution |
| Mesh `rdma/mesh-metal4.m:210` | block | mesh_metal4_commit | passed to addFeedbackHandler: | completion |
| Mesh `rdma/mesh-ports.c:478` | block | ports_find | passed to dispatch_async | preparation |
| Mesh `rdma/mesh-records.c:165` | named function | map_end | registered with mesh_graph_reach | preparation |
| Mesh `rdma/mesh-records.c:369` | named function | island_parent | registered with mesh_graph_rows | preparation |
| Mesh `rdma/mesh-records.c:881` | named function | seen_group | registered with qsort and bsearch | preparation |
| Mesh `rdma/mesh-torch.mm:65` | block | after_torch | passed to done->notify | completion |
| Mesh `rdma/mesh-torch.mm:87` | block | mesh_torch_publish | passed to after_torch | completion |
| Mesh `rdma/mesh-torch.mm:104` | block | mesh_torch_retire | passed to after_torch | completion |
| Mesh `rdma/mesh-torch.mm:120` | block | mesh_torch_end | passed to after_torch | completion |
| Mesh `swift/Bounds.swift:114` | Swift closure | - | passed to reduce | compilation/construction |
| Mesh `swift/Bounds.swift:116` | Swift closure | - | passed to map | compilation/construction |
| Metal `metal_archive.mm:34` | named function | ReleaseArchiveBytes | registered with mesh_graph_relation_load | completion (release) |
| Metal `metal_archive.mm:440` | lambda | UpstreamModule | passed to tint::core::ir::Traverse | preparation |
| Metal `metal_archive.mm:1139` | named function | metal_ir_archive_free | registered outward by obsolete rdma/mesh_compile.py RELEASE | completion (release) |
| Metal `metal_capture.m:343` | block | MetalStandIns | passed to addObject:MetalStateWord | preparation |
| Metal `metal_capture.m:556` | named function | MetalRecorderActorsFree | registered with weakref.finalize by compiler/mesh_compiler/native.py; the active definition (Makefile RECORDING_SRCS) | completion (release) |
| Metal `metal_capture.m:706` | block | MetalPreparedHandles | passed to deallocator: | completion |
| Metal `metal_capture.m:1037` | block | MetalReplayPlan | passed to MetalReplaySegments | preparation |
| Metal `metal_capture.m:1516` | block | MetalReplayOptimize | passed to MetalReplaySegments | preparation |
| Metal `metal_capture.m:1560` | block | MetalBufferBorrow | passed to deallocator: | completion |
| Metal `metal_clone.mm:109` | lambda | Register | passed to ir::Traverse | compilation/construction |
| Metal `metal_clone.mm:141` | lambda | Retype | passed to ir::Traverse | compilation/construction |
| Metal `metal_clone.mm:154` | lambda | Retype | passed to control->ForeachBlock | compilation/construction |
| Metal `metal_clone.mm:181` | lambda | AccessMode | passed to ->ForEachUseUnsorted | compilation/construction |
| Metal `metal_clone.mm:203` | lambda | (file scope) | passed to ir::Traverse | compilation/construction |
| Metal `metal_clone.mm:219` | lambda | (file scope) | passed to remapped | compilation/construction |
| Metal `metal_ir.mm:529` | lambda | Compose | stored as c.offset | compilation/construction |
| Metal `metal_ir.mm:559` | lambda | ComposeGroup | stored as c.offset | compilation/construction |
| Metal `metal_ir.mm:651` | lambda | Normalize | passed to std::stable_sort | compilation/construction |
| Metal `metal_ir.mm:807` | named function | Undecided | registered with mesh_graph_relation_views | compilation/construction |
| Metal `metal_ir.mm:932` | lambda | OperationConflicts | passed to Hazards | compilation/construction |
| Metal `metal_ir.mm:942` | lambda | Conflict | passed to Hazards | compilation/construction |
| Metal `metal_ir.mm:998` | lambda | RegionForest | passed to mesh_graph_climb | compilation/construction |
| Metal `metal_ir.mm:1012` | lambda | RegionForest | passed to mesh_graph_climb | compilation/construction |
| Metal `metal_ir.mm:1023` | lambda | RegionForest | passed to mesh_graph_climb | compilation/construction |
| Metal `metal_ir.mm:1040` | lambda | RegionAbove | passed to mesh_graph_climb | compilation/construction |
| Metal `metal_ir.mm:1083` | lambda | SwitchArguments | passed to std::none_of | compilation/construction |
| Metal `metal_ir.mm:1113` | lambda | InsertKernels | passed to Renumber | compilation/construction |
| Metal `metal_ir.mm:1128` | lambda | EraseKernel | passed to Renumber | compilation/construction |
| Metal `metal_ir.mm:1141` | lambda | Internal | passed to std::any_of | compilation/construction |
| Metal `metal_ir.mm:1145` | lambda | Internal | passed to std::none_of | compilation/construction |
| Metal `metal_ir.mm:1385` | lambda | Footprints | passed to std::any_of | compilation/construction |
| Metal `metal_ir.mm:1510` | lambda | Decode | passed to Dequantize | compilation/construction |
| Metal `metal_ir.mm:1662` | lambda | Addresses | passed to b.Append | compilation/construction |
| Metal `metal_ir.mm:1665` | lambda | Addresses | passed to Address | compilation/construction |
| Metal `metal_ir.mm:1748` | lambda | Store | passed to Encode | compilation/construction |
| Metal `metal_ir.mm:1755` | lambda | Stored | passed to Encode | compilation/construction |
| Metal `metal_ir.mm:1848` | lambda | Logical | passed to std::find_if | compilation/construction |
| Metal `metal_ir.mm:1881` | lambda | LogicalRead | passed to Dequantize | compilation/construction |
| Metal `metal_ir.mm:1894` | lambda | Accesses | passed to Load | compilation/construction |
| Metal `metal_ir.mm:1900` | lambda | Accesses | passed to StoreRaw | compilation/construction |
| Metal `metal_ir.mm:1903` | lambda | Accesses | passed to StoreRaw | compilation/construction |
| Metal `metal_ir.mm:1927` | lambda | Counts | passed to ir::Traverse | compilation/construction |
| Metal `metal_ir.mm:2177` | lambda | CanonicalWork | same, passed by name at lines 2152 | compilation/construction |
| Metal `metal_ir.mm:2268` | lambda | Actor | passed to std::remove_if | compilation/construction |
| Metal `metal_ir.mm:2373` | named function | ReleaseDecidedInputs | registered with mesh_graph_relation_decided | completion (release) |
| Metal `metal_ir.mm:2384` | named function | Decide | registered with mesh_graph_relation_decided | compilation/construction |
| Metal `metal_ir.mm:2511` | named function | CutDegree | stored in a mesh_graph_rows_view | compilation/construction |
| Metal `metal_ir.mm:2516` | named function | CutHead | stored in a mesh_graph_rows_view | compilation/construction |
| Metal `metal_ir.mm:2585` | named function | ReplayDegree | stored in a mesh_graph_rows_view | compilation/construction |
| Metal `metal_ir.mm:2590` | named function | ReplayHead | stored in a mesh_graph_rows_view | compilation/construction |
| Metal `metal_ir.mm:3021` | lambda | Binary64 | passed to ir::Traverse | compilation/construction |
| Metal `metal_ir.mm:3079` | lambda | Binary64 | passed to control->ForeachBlock | compilation/construction |
| Metal `metal_ir.mm:3095` | lambda | Binary64 | passed to b.InsertBefore | compilation/construction |
| Metal `metal_ir.mm:3111` | lambda | Binary64 | passed to Splice | compilation/construction |
| Metal `metal_ir.mm:3190` | lambda | Annotated | passed to control->ForeachBlock | compilation/construction |
| Metal `metal_ir.mm:3419` | lambda | SwitchArm | passed to mesh_graph_climb | compilation/construction |
| Metal `metal_ir.mm:3439` | lambda | SwitchBodies | passed to std::any_of | compilation/construction |
| Metal `metal_ir.mm:3510` | named function | metal_ir_program_free | registered as a std::unique_ptr deleter (metal_archive.mm) | completion (release) |
| Metal `metal_ir.mm:3524` | named function | Released | stored as a program owner release | completion (release) |
| Metal `metal_ir.mm:3642` | lambda | AppendSource | passed to clone | compilation/construction |
| Metal `metal_ir.mm:4211` | lambda | ranges | stored as argumentAt | compilation/construction |
| Metal `metal_ir.mm:4216` | lambda | ranges | stored as valueAt | compilation/construction |
| Metal `metal_ir.mm:4605` | lambda | Loops | passed to tint::core::ir::Traverse | compilation/construction |
| Metal `metal_ir.mm:4664` | lambda | Walk | passed to control->ForeachBlock | compilation/construction |
| Metal `metal_ir.mm:4843` | lambda | (file scope) | stored as x.argumentAt | compilation/construction |
| Metal `metal_ir.mm:4845` | lambda | (file scope) | stored as x.valueAt | compilation/construction |
| Metal `metal_ir.mm:4867` | lambda | (file scope) | passed to tint::core::ir::Traverse | compilation/construction |
| Metal `metal_ir.mm:5027` | named function | metal_ir_binary64_free | registered with weakref.finalize by compiler/mesh_compiler/metal.py Binary64 | completion (release) |
| Metal `metal_ir.mm:5937` | lambda | metal_ir_extract_if | passed to b.Append | compilation/construction |
| Metal `metal_ir.mm:5939` | lambda | metal_ir_extract_if | passed to b.Append | compilation/construction |
| Metal `metal_ir.mm:5955` | lambda | metal_ir_scatter | passed to Quantize | compilation/construction |
| Metal `metal_ir.mm:5966` | lambda | metal_ir_scatter | passed to b.Append | compilation/construction |
| Metal `metal_ir.mm:5971` | lambda | metal_ir_scatter | passed to b.Append | compilation/construction |
| Metal `metal_ir.mm:6744` | lambda | metal_ir_value_physical | passed to Physical | compilation/construction |
| Metal `metal_ir.mm:6888` | lambda | Touched | passed to Paged | compilation/construction |
| Metal `metal_ir.mm:7028` | lambda | metal_ir_place | passed to Physical | compilation/construction |
| Metal `metal_ir.mm:7372` | lambda | metal_ir_operation_axis | same, passed by name at lines 7314 | compilation/construction |
| Metal `metal_lower.h:101` | lambda | When | passed to b.Append | compilation/construction |
| Metal `metal_lower.h:105` | lambda | When | passed to b.Append | compilation/construction |
| Metal `metal_lower.h:110` | lambda | For | passed to b.LoopRange | compilation/construction |
| Metal `metal_lower.mm:144` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:146` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:148` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:150` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:152` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:154` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:156` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:158` | lambda | Builder::One | passed to tint::Switch | compilation/construction |
| Metal `metal_lower.mm:177` | lambda | Builder::Storage | passed to b.Append | compilation/construction |
| Metal `metal_lower.mm:202` | lambda | Builder::Storage | passed to b.Append | compilation/construction |
| Metal `metal_lower.mm:222` | lambda | Builder::Workgroup | passed to b.Append | compilation/construction |
| Metal `metal_lower.mm:269` | lambda | Builder::Read | passed to metal_ir::Load | compilation/construction |
| Metal `metal_lower.mm:287` | lambda | Builder::Write | storage, passed by name at lines 275 | compilation/construction |
| Metal `metal_lower.mm:295` | lambda | Builder::Write | passed to metal_ir::Store | compilation/construction |
| Metal `metal_lower.mm:297` | lambda | Builder::Write | passed to metal_ir::Store | compilation/construction |
| Metal `metal_lower.mm:351` | lambda | Builder::Inline | passed to Splice | compilation/construction |
| Metal `metal_lower.mm:369` | lambda | Builder::Inline | passed to Splice | compilation/construction |
| Metal `metal_lower.mm:411` | lambda | Builder::View | passed to b.Append | compilation/construction |
| Metal `metal_lower.mm:653` | lambda | Contiguity | passed to std::stable_sort | compilation/construction |
| Metal `metal_lower.mm:938` | lambda | Member | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:940` | lambda | Member | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:992` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:997` | lambda | Member | passed to e.For | compilation/construction |
| Metal `metal_lower.mm:999` | lambda | Member | passed to point | compilation/construction |
| Metal `metal_lower.mm:1008` | lambda | Member | passed to e.For | compilation/construction |
| Metal `metal_lower.mm:1041` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1047` | lambda | Member | passed to e.For | compilation/construction |
| Metal `metal_lower.mm:1054` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1114` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1141` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1159` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1165` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1172` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1181` | lambda | Member | passed to e.When | compilation/construction |
| Metal `metal_lower.mm:1243` | lambda | Before | in, passed by name at lines 1210 | compilation/construction |
| Metal `metal_lower.mm:1246` | lambda | Before | passed to std::any_of | compilation/construction |
| Metal `metal_lower.mm:1308` | lambda | Construct | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:1350` | lambda | Construct | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:1528` | lambda | TaskLibrary | passed to std::shared_ptr | compilation/construction |
| Metal `metal_lower.mm:2040` | lambda | CompileTask | passed to std::find_if | compilation/construction |
| Metal `metal_lower.mm:2396` | lambda | LowerProgram | passed to std::none_of | compilation/construction |
| Metal `metal_lower.mm:2458` | lambda | PrepareConstants | passed to std::none_of | compilation/construction |
| Metal `metal_lower.mm:2468` | lambda | PrepareConstants | passed to into.b.Append | compilation/construction |
| Metal `metal_lower.mm:2470` | lambda | PrepareConstants | passed to Splice | compilation/construction |
| Metal `metal_lower.mm:2481` | lambda | PrepareConstants | passed to into.b.Append | compilation/construction |
| Metal `metal_lower.mm:2505` | lambda | PrepareConstants | passed to mesh_graph_rows | compilation/construction |
| Metal `metal_lower.mm:2512` | lambda | PrepareConstants | endpoint, passed by name at lines 2530 | compilation/construction |
| Metal `metal_lower.mm:2524` | lambda | PrepareConstants | passed to std::any_of | compilation/construction |
| Metal `metal_lower.mm:2534` | lambda | PrepareConstants | passed to std::none_of | compilation/construction |
| Metal `metal_lower.mm:2552` | lambda | PrepareConstants | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:2582` | lambda | PrepareConstants | passed to mesh_graph_preorder | compilation/construction |
| Metal `metal_lower.mm:2610` | lambda | PrepareConstants | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:2621` | lambda | PrepareConstants | passed to e.For | compilation/construction |
| Metal `metal_lower.mm:2686` | lambda | PrepareConstants | passed to std::erase_if | compilation/construction |
| Metal `metal_lower.mm:2709` | lambda | PrepareConstants | passed to std::remove_if | compilation/construction |
| Metal `metal_lower.mm:2713` | lambda | PrepareConstants | passed to std::remove_if | compilation/construction |
| Metal `metal_lower.mm:2716` | lambda | PrepareConstants | passed to std::none_of | compilation/construction |
| Metal `metal_lower.mm:2720` | lambda | PrepareConstants | passed to std::all_of | compilation/construction |
| Metal `metal_lower.mm:2861` | lambda | Wrap | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:2919` | lambda | Step | passed to e.b.Append | compilation/construction |
| Metal `metal_lower.mm:3170` | lambda | ConstructKernels | bound, passed by name at lines 3110 | compilation/construction |
| Metal `metal_lower.mm:3182` | lambda | ConstructKernels | passed to std::any_of | compilation/construction |
| Metal `metal_lower.mm:3316` | block | Emit | passed to dispatch_apply | compilation/construction |
| Metal `metal_lower.mm:3359` | block | Emit | passed to dispatch_apply | compilation/construction |
| Metal `metal_native.mm:31` | block | MetalNativeCores | passed to dispatch_once | preparation |
| Metal `metal_native.mm:76` | lambda | OperandGraph | passed to ir::Traverse | compilation/construction |
| Metal `metal_native.mm:314` | lambda | MapHost | passed to std::any_of | compilation/construction |
| Metal `metal_native.mm:437` | block | Family | passed to dispatch_once | compilation/construction |
| Metal `metal_native.mm:469` | lambda | Relayout | slice, passed by name at lines 482 | compilation/construction |
| Metal `metal_native.mm:488` | lambda | Relayout | passed to dispatch_apply_f | compilation/construction |
| Metal `metal_native.mm:590` | lambda | Specialize | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:604` | lambda | Activation | passed to Splice | compilation/construction |
| Metal `metal_native.mm:712` | lambda | ProjectionMember | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:728` | lambda | ProjectionMember | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:754` | lambda | ProjectionMember | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:763` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:799` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:814` | lambda | ProjectionMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:820` | lambda | ProjectionMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:842` | lambda | ProjectionMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:848` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:887` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:897` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:914` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:922` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:930` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:938` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:947` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:953` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:956` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:964` | lambda | ProjectionMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:1056` | lambda | GemmMember | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:1082` | lambda | GemmMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:1085` | lambda | GemmMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:1098` | lambda | GemmMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1100` | lambda | GemmMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1119` | lambda | GemmMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1130` | lambda | GemmMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1136` | lambda | GemmMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:1177` | lambda | GemmMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1180` | lambda | GemmMember | passed to e.When | compilation/construction |
| Metal `metal_native.mm:1233` | lambda | HostMember | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:1246` | lambda | HostMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1248` | lambda | HostMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1270` | lambda | HostMember | passed to each | compilation/construction |
| Metal `metal_native.mm:1280` | lambda | HostMember | passed to each | compilation/construction |
| Metal `metal_native.mm:1455` | lambda | LapackMember | passed to e.b.Append | compilation/construction |
| Metal `metal_native.mm:1474` | lambda | LapackMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1476` | lambda | LapackMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1478` | lambda | LapackMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1486` | lambda | LapackMember | passed to each | compilation/construction |
| Metal `metal_native.mm:1494` | lambda | LapackMember | passed to e.For | compilation/construction |
| Metal `metal_native.mm:1574` | lambda | LapackMember | passed to each | compilation/construction |
| Metal `metal_neural.mm:66` | named function | Release | stored as a program owner release | completion (release) |
| Metal `metal_neural.mm:188` | block | Invoke | passed to handler: | execution |
| Metal `metal_neural.mm:202` | block | Invoke | passed to handler: | execution |
| Metal `metal_neural.mm:349` | block | EmitNeural | passed to completionHandler: | compilation/construction (completion handler) |
| Metal `metal_program.h:214` | block | NativeNotify | passed to block: | completion |
| Metal `metal_program.m:403` | block | MetalProgramCommit | passed to addFeedbackHandler: | completion |
| Metal `metal_program.m:410` | block | MetalProgramCommit | passed to dispatch_async | completion |
| Metal `metal_program.m:420` | block | MetalOptimizeCommit | passed to addFeedbackHandler: | completion |
| Metal `metal_program.m:427` | block | MetalOptimizeCommit | passed to dispatch_async | completion |
| Metal `metal_program.m:499` | named function | MetalRelationOwned | registered with mesh_graph_relation_views and mesh_graph_signature_scopes | completion (release) |
| Metal `metal_program.m:502` | named function | MetalArchiveReleased | registered with mesh_graph_relation_load | completion (release) |
| Metal `metal_program.m:1399` | block | MetalRepeatRun | passed to enumerateIndexesUsingBlock: | execution |
| Metal `metal_program.m:1921` | block | MetalSentWait | passed to MetalReady | execution |
| Metal `metal_program.m:1926` | block | MetalSentWait | passed to MetalReady | execution |
| Metal `metal_program.m:1936` | block | MetalNativeWait | passed to MetalReady | execution |
| Metal `metal_program.m:1939` | block | MetalNativeWait | passed to MetalReady | execution |
| Metal `metal_program.m:2012` | named function | MetalRunCompletion | stored as the run hooks .completion | completion |
| Metal `metal_program.m:2017` | named function | MetalRunCommit | stored as the run hooks .commit | execution |
| Metal `metal_program.m:2040` | block | MetalExternalRelease | passed to block: | completion |
| Metal `metal_program.m:2433` | named function | MetalScopesReleased | registered with mesh_graph_scopes_create | completion (release) |
| Metal `metal_program.m:2616` | named function | MetalSourcesReleased | registered with mesh_graph_scopes_bind | completion (release) |
| Metal `metal_recording.m:533` | block | MetalStandIns | passed to addObject:MetalStateWord | preparation |
| Metal `metal_recording.m:758` | named function | MetalRecorderActorsFree | the retained inactive alternative of metal_capture.m's, which native.py's weakref.finalize reaches: metal_recording.m is outside Makefile RECORDING_SRCS | completion (release) |
| Metal `metal_recording.m:1370` | block | MetalReplayOptimize | passed to MetalReplaySegments | preparation |
| Metal `metal_recording.m:1410` | block | MetalBufferBorrow | passed to deallocator: | completion |
| Metal `metal_recording.m:1553` | block | MetalNativeICB | passed to MetalReplaySegments | preparation |
| Metal `metal_recording.m:1574` | block | MetalNativeICB | passed to enumerateIndexesUsingBlock: | preparation |
| Metal `metal_recording.m:1795` | block | MetalProgramCommit | passed to addFeedbackHandler: | completion |
| Metal `metal_recording.m:1802` | block | MetalProgramCommit | passed to dispatch_async | completion |
| Metal `metal_recording.m:1812` | block | MetalOptimizeCommit | passed to addFeedbackHandler: | completion |
| Metal `metal_recording.m:1819` | block | MetalOptimizeCommit | passed to dispatch_async | completion |
| Metal `metal_recording.m:1891` | named function | MetalRelationOwned | registered with mesh_graph_relation_views and mesh_graph_signature_scopes | completion (release) |
| Metal `metal_recording.m:1894` | named function | MetalArchiveReleased | registered with mesh_graph_relation_load | completion (release) |
| Metal `metal_recording.m:2965` | block | MetalSentWait | passed to MetalReady | execution |
| Metal `metal_recording.m:2970` | block | MetalSentWait | passed to MetalReady | execution |
| Metal `metal_recording.m:2980` | block | MetalNativeWait | passed to MetalReady | execution |
| Metal `metal_recording.m:2983` | block | MetalNativeWait | passed to MetalReady | execution |
| Metal `metal_recording.m:3046` | named function | MetalRunCompletion | stored as the run hooks .completion | completion |
| Metal `metal_recording.m:3051` | named function | MetalRunCommit | stored as the run hooks .commit | execution |
| Metal `metal_recording.m:3074` | block | MetalExternalRelease | passed to block: | completion |
| Metal `metal_transform.mm:121` | lambda | Fold | passed to control->ForeachBlock | compilation/construction |
| Metal `metal_transform.mm:150` | lambda | Prune | passed to control->ForeachBlock | compilation/construction |
| Metal `metal_transform.mm:238` | named function | Any | registered with Conflict | compilation/construction |
| Metal `metal_transform.mm:240` | named function | Writes | registered with Conflict | compilation/construction |
| Metal `metal_transform.mm:242` | named function | Reads | registered with Conflict | compilation/construction |
| Metal `metal_transform.mm:254` | lambda | Discard | passed to p.regions.functions.EraseIf | compilation/construction |
| Metal `metal_transform.mm:262` | lambda | Spliced | passed to Splice | compilation/construction |
| Metal `metal_transform.mm:287` | lambda | Element | passed to Stored | compilation/construction |
| Metal `metal_transform.mm:299` | lambda | Clipped | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:409` | lambda | Fuse | passed to b.Append | compilation/construction |
| Metal `metal_transform.mm:498` | lambda | Inline | written, passed by name at lines 492,493 | compilation/construction |
| Metal `metal_transform.mm:516` | lambda | Inline | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:520` | lambda | Inline | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:574` | lambda | Epilogue | written, passed by name at lines 566,568 | compilation/construction |
| Metal `metal_transform.mm:591` | lambda | Epilogue | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:597` | lambda | Epilogue | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:604` | lambda | Epilogue | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:619` | lambda | Epilogue | reads, passed by name at lines 603,606 | compilation/construction |
| Metal `metal_transform.mm:621` | lambda | Epilogue | writes, passed by name at lines 603,606 | compilation/construction |
| Metal `metal_transform.mm:699` | lambda | Epilogue | gathered, passed by name at lines 696 | compilation/construction |
| Metal `metal_transform.mm:705` | lambda | Epilogue | passed to b.Append | compilation/construction |
| Metal `metal_transform.mm:867` | lambda | Partitioned | passed to std::find_if | compilation/construction |
| Metal `metal_transform.mm:874` | lambda | Partitioned | passed to std::remove_if | compilation/construction |
| Metal `metal_transform.mm:878` | lambda | Partitioned | passed to std::sort | compilation/construction |
| Metal `metal_transform.mm:906` | lambda | Decodes | passed to std::sort | compilation/construction |
| Metal `metal_transform.mm:922` | lambda | Decodes | passed to std::sort | compilation/construction |
| Metal `metal_transform.mm:1039` | lambda | Admitted | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:1179` | lambda | Proven | passed to std::any_of | compilation/construction |
| Metal `metal_transform.mm:1215` | lambda | Proven | passed to std::equal | compilation/construction |
| Metal `metal_transform.mm:1234` | lambda | Proven | passed to std::any_of | compilation/construction |
| Metal `metal_transport.mm:427` | lambda | Obligate | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:458` | lambda | Compact | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:462` | lambda | Compact | passed to e.For | compilation/construction |
| Metal `metal_transport.mm:464` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:471` | lambda | Compact | passed to e.For | compilation/construction |
| Metal `metal_transport.mm:479` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:482` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:487` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:489` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:494` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:502` | lambda | Compact | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:844` | lambda | Transport | passed to later.module->functions.EraseIf | compilation/construction |
| Metal `metal_transport.mm:979` | lambda | Acquiring | passed to e.For | compilation/construction |
| Metal `metal_transport.mm:1001` | lambda | Acquiring | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1014` | lambda | Acquiring | passed to e.For | compilation/construction |
| Metal `metal_transport.mm:1023` | lambda | Acquiring | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1028` | lambda | Acquiring | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1032` | lambda | Acquiring | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:1034` | lambda | Acquiring | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:1039` | lambda | Acquiring | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1041` | lambda | Acquiring | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1043` | lambda | Acquiring | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1058` | lambda | Acquiring | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:1075` | lambda | Releasing | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1079` | lambda | Releasing | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1086` | lambda | Releasing | passed to e.When | compilation/construction |
| Metal `metal_transport.mm:1118` | lambda | Native | stored as o.group | compilation/construction |
| Metal `metal_transport.mm:1120` | lambda | Native | stored as o.statusOffset | compilation/construction |
| Metal `metal_transport.mm:1122` | lambda | Native | stored as o.range | compilation/construction |
| Metal `metal_transport.mm:1130` | lambda | Native | stored as o.klass | compilation/construction |
| Metal `metal_transport.mm:1139` | lambda | Native | stored as o.member | compilation/construction |
| Metal `metal_transport.mm:1147` | lambda | Native | stored as o.records | compilation/construction |
| Metal `metal_transport.mm:1149` | lambda | Native | stored as o.total | compilation/construction |
| Metal `metal_transport.mm:1151` | lambda | Native | stored as o.release | compilation/construction |
| Metal `metal_transport.mm:1173` | lambda | Acquire | passed to Acquiring | compilation/construction |
| Metal `metal_transport.mm:1216` | lambda | UpstreamProtocol | stored as o.range | compilation/construction |
| Metal `metal_transport.mm:1218` | lambda | UpstreamProtocol | stored as o.group | compilation/construction |
| Metal `metal_transport.mm:1220` | lambda | UpstreamProtocol | stored as o.statusOffset | compilation/construction |
| Metal `metal_transport.mm:1222` | lambda | UpstreamProtocol | stored as o.klass | compilation/construction |
| Metal `metal_transport.mm:1227` | lambda | UpstreamProtocol | stored as o.member | compilation/construction |
| Metal `metal_transport.mm:1231` | lambda | UpstreamProtocol | stored as o.records | compilation/construction |
| Metal `metal_transport.mm:1233` | lambda | UpstreamProtocol | stored as o.total | compilation/construction |
| Metal `metal_transport.mm:1235` | lambda | UpstreamProtocol | stored as o.release | compilation/construction |
| Metal `metal_transport.mm:1318` | lambda | UpstreamProtocol | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:1326` | lambda | UpstreamProtocol | passed to Acquiring | compilation/construction |
| Metal `metal_transport.mm:1331` | lambda | UpstreamProtocol | passed to Acquiring | compilation/construction |
| Metal `metal_transport.mm:1349` | lambda | UpstreamProtocol | passed to viewed | compilation/construction |
| Metal `metal_transport.mm:1368` | lambda | UpstreamProtocol | passed to e.b.Append | compilation/construction |
| Metal `metal_transport.mm:1578` | lambda | ModulePass | passed to b.Append | compilation/construction |
| Metal `metal_transport.mm:1590` | lambda | ModulePass | passed to typed | compilation/construction |
| Metal `metal_transport.mm:1595` | lambda | ModulePass | passed to typed | compilation/construction |
| Metal `metal_transport.mm:1597` | lambda | ModulePass | passed to typed | compilation/construction |
| Metal `metal_transport.mm:1602` | lambda | ModulePass | passed to typed | compilation/construction |
| Metal `metal_transport.mm:1604` | lambda | ModulePass | passed to typed | compilation/construction |
| Metal `metal_transport.mm:1666` | lambda | ModulePass | passed to b.Append | compilation/construction |
| Metal `metal_transport.mm:1673` | lambda | ModulePass | passed to b.Append | compilation/construction |
| Metal `provider/litert_mesh.cc:110` | named function | HolderRelease | registered with metal_ir_value_data | completion (release) |
| Metal `provider/litert_mesh.cc:413` | lambda | Emit | passed to EmitAll | compilation/construction |
| Metal `provider/litert_mesh.cc:522` | lambda | Reduce | passed to EmitAll | compilation/construction |
| Metal `provider/litert_mesh.cc:537` | lambda | Copy | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:551` | lambda | Fill | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:681` | lambda | Convolve | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:685` | lambda | Convolve | passed to Map | compilation/construction |
| Metal `provider/litert_mesh.cc:721` | lambda | Pool | passed to EmitAll | compilation/construction |
| Metal `provider/litert_mesh.cc:731` | lambda | Pool | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:754` | lambda | Pool | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:757` | lambda | Pool | passed to Map | compilation/construction |
| Metal `provider/litert_mesh.cc:804` | lambda | Deconvolve | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:808` | lambda | Deconvolve | passed to Map | compilation/construction |
| Metal `provider/litert_mesh.cc:867` | lambda | Softmax | passed to Reduce | compilation/construction |
| Metal `provider/litert_mesh.cc:879` | lambda | Softmax | passed to EmitAll | compilation/construction |
| Metal `provider/litert_mesh.cc:892` | lambda | Softmax | passed to EmitAll | compilation/construction |
| Metal `provider/litert_mesh.cc:926` | lambda | MatMul | passed to Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:930` | lambda | MatMul | passed to Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1131` | lambda | Offset | passed to l_.Computed | compilation/construction |
| Metal `provider/litert_mesh.cc:1333` | lambda | Elementwise | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1370` | lambda | Reduce | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1375` | lambda | Reduce | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1401` | lambda | ArgExtreme | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1469` | lambda | RngBits | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1497` | lambda | RngBits | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1542` | lambda | Random | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1572` | lambda | Random | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1589` | lambda | Padded | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1668` | lambda | ndow | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1671` | lambda | ndow | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1745` | lambda | Gather | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1765` | lambda | GatherNd | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1798` | lambda | ScatterNd | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:1929` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1940` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:1949` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2044` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2053` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2113` | lambda | Lower | passed to l_.Reduce | compilation/construction |
| Metal `provider/litert_mesh.cc:2115` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2123` | lambda | Lower | passed to l_.Reduce | compilation/construction |
| Metal `provider/litert_mesh.cc:2134` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2136` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2174` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2208` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2258` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2270` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2294` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2317` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2341` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2357` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2361` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2383` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2399` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2410` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2470` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2480` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2486` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2511` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2539` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2584` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2609` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:2621` | lambda | Lower | passed to l_.Emit | compilation/construction |
| Metal `provider/litert_mesh.cc:2663` | lambda | Lower | passed to l_.Map | compilation/construction |
| Metal `provider/litert_mesh.cc:3020` | lambda | (file scope) | passed to std::sort | compilation/construction |
| Metal `provider/litert_mesh.cc:3723` | lambda | Publish | boundary, passed by name at lines 3697 | compilation/construction |
| Metal `tools/e2b_replay.m:45` | named function | model_retain | registered with LiteRtMeshExport | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:47` | named function | model_release | registered with LiteRtMeshExport | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:93` | block | program_actors | passed to deallocator: | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:174` | named function | feedback_record | registered with MetalProgramFeedback | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:192` | named function | stop_replay | registered with dispatch_source_set_event_handler_f | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:202` | named function | span_start | registered with MetalProgramRun | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:208` | named function | span_started | registered with MetalProgramRun | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:212` | named function | adapter_ready | stored as a WGPU callback | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:218` | named function | device_error | stored as a WGPU callback | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:696` | named function | e2b_placement | registered with e2b_ir | compilation/construction (Mesh tool) |
| Metal `tools/e2b_replay.m:716` | named function | e2b_unit_weights | registered with e2b_ir | compilation/construction (Mesh tool) |
| Metal `tools/e2b_replay.m:1266` | block | main | role, passed by name at lines 1262 | execution (Mesh tool) |
| Metal `tools/e2b_replay.m:1641` | block | main | passed to host_elements | execution (Mesh tool) |
| Metal `tools/native_bench.m:47` | named function | Start | registered with MetalProgramRun | execution (Mesh tool) |
| Metal `tools/native_bench.m:49` | named function | Started | registered with MetalProgramRun | execution (Mesh tool) |
| Metal `tools/native_bench.m:61` | named function | ArchiveReleased | registered with metal_ir_archive_load and as a release | completion (release) |
| Metal `tools/native_bench.m:65` | block | ActorsHeld | passed to deallocator: | execution (Mesh tool) |
| Metal `tools/native_bench.m:127` | named function | NativeReleased | registered with metal_ir_archive_load and as a release | completion (release) |
| Metal `tools/native_bench.m:224` | named function | Feedback | registered with MetalProgramFeedback | execution (Mesh tool) |
| Metal `tools/native_bench.m:459` | block | Projection | passed to dispatch_apply | execution (Mesh tool) |
| Metal `tools/primitive_constants.m:184` | block | Timed | passed to addFeedbackHandler: | execution (Mesh tool) |
| Metal `tools/primitive_constants.m:210` | block | Loop | passed to Timed | execution (Mesh tool) |
| Metal `tools/primitive_constants.m:226` | block | Staged | passed to Timed | execution (Mesh tool) |
| Metal `tools/primitive_constants.m:260` | block | Commands | passed to Timed | execution (Mesh tool) |
| LiteRT `ml_drift_delegate/delegate/delegate_kernel.cc:361` | provider callback handoff | ABSL_RETURN_IF_ERROR(::ml_drift::GraphToGpuModel( | GraphToGpuModel supplied OpSelector | compilation/construction |
| LiteRT `ml_drift_delegate/delegate/delegate_kernel.cc:1348` | provider callback handoff | ABSL_RETURN_IF_ERROR(::ml_drift::ir::IrModelToGpuModel( | IrModelToGpuModel supplied IrModelOpSelector | compilation/construction |
| LiteRT `ml_drift_delegate/delegate/delegate_webgpu.cc:523` | named function | Invoke | TfLiteRegistration.invoke | execution |
| LiteRT `ml_drift_delegate/delegate/delegate_webgpu.cc:570` | named function | DelegatePrepare | TfLiteDelegate.Prepare | preparation |
| LiteRT `tflite/kernels/comparisons.cc:173` | provider callback handoff | reference_ops::BroadcastComparison4DSlowStringImpl( | BroadcastComparison4DSlowStringImpl predicate | execution |
| LiteRT `tflite/kernels/comparisons.cc:179` | provider callback handoff | reference_ops::ComparisonStringImpl(opname, | ComparisonStringImpl predicate | execution |
| LiteRT `tflite/kernels/comparisons.cc:417` | named function | EqualEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/comparisons.cc:422` | named function | EqualPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/comparisons.cc:428` | named function | NotEqualEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/comparisons.cc:433` | named function | NotEqualPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/comparisons.cc:439` | named function | LessEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/comparisons.cc:444` | named function | LessPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/comparisons.cc:451` | named function | LessEqualEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/comparisons.cc:457` | named function | LessEqualPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/comparisons.cc:464` | named function | GreaterEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/comparisons.cc:469` | named function | GreaterPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/comparisons.cc:476` | named function | GreaterEqualEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/comparisons.cc:482` | named function | GreaterEqualPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/logical.cc:44` | named function | Init | TfLiteRegistration.init | preparation |
| LiteRT `tflite/kernels/logical.cc:132` | named function | LogicalOrEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/logical.cc:139` | named function | LogicalOrPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/logical.cc:148` | named function | LogicalAndEval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/logical.cc:155` | named function | LogicalAndPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/logical.h:10` | named function | LogicalOr | reference_ops::BinaryFunction/BroadcastBinaryFunction4DSlow predicate | execution |
| LiteRT `tflite/kernels/logical.h:13` | named function | LogicalAnd | reference_ops::BinaryFunction/BroadcastBinaryFunction4DSlow predicate | execution |
| LiteRT `tflite/kernels/mesh_selected.cc:8` | named function | Init | TfLiteRegistration.init | preparation |
| LiteRT `tflite/kernels/mesh_selected.cc:14` | named function | Free | TfLiteRegistration.free | completion (release) |
| LiteRT `tflite/kernels/reshape.cc:168` | named function | Prepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/reshape.cc:187` | named function | Eval | TfLiteRegistration.invoke | execution |
| LiteRT `tflite/kernels/reshape.cc:245` | named function | Init | TfLiteRegistration.init | preparation |
| LiteRT `tflite/kernels/select.cc:50` | named function | SelectInit | TfLiteRegistration.init | preparation |
| LiteRT `tflite/kernels/select.cc:67` | named function | SelectPrepare | TfLiteRegistration.prepare | preparation |
| LiteRT `tflite/kernels/select.cc:203` | named function | SelectEval | TfLiteRegistration.invoke | execution |
| MLDrift `ml_drift/common/merge_nodes.h:39` | supplied-object virtual method | Width | GPUOperation supplied GpuSpatialTensor interface via SetSrc/SetDst | compilation/construction |
| MLDrift `ml_drift/common/merge_nodes.h:41` | supplied-object virtual method | Height | GPUOperation supplied GpuSpatialTensor interface via SetSrc/SetDst | compilation/construction |
| MLDrift `ml_drift/common/merge_nodes.h:43` | supplied-object virtual method | Depth | GPUOperation supplied GpuSpatialTensor interface via SetSrc/SetDst | compilation/construction |
| MLDrift `ml_drift/common/merge_nodes.h:45` | supplied-object virtual method | Channels | GPUOperation supplied GpuSpatialTensor interface via SetSrc/SetDst | compilation/construction |
| MLDrift `ml_drift/common/merge_nodes.h:47` | supplied-object virtual method | Slices | GPUOperation supplied GpuSpatialTensor interface via SetSrc/SetDst | compilation/construction |
| MLDrift `ml_drift/common/merge_nodes.h:49` | supplied-object virtual method | Batch | GPUOperation supplied GpuSpatialTensor interface via SetSrc/SetDst | compilation/construction |
| MLDrift `ml_drift/webgpu/arguments.cc:271` | provider callback handoff | uniform_buffer_creator->CreateUniformBuffer (first Init overload) | WebGpuArguments::Init supplied UniformBufferCreator | preparation |
| XNNPACK `src/operator-run.c:2078` | provider callback handoff | parallelize(threadpool, run_task, context, __VA_ARGS__, flags); | pthreadpool_parallelize_* task operand (XNN_PARALLELIZE) | execution |
| XNNPACK `src/subgraph/batch-matrix-multiply.c:512` | named function | setup_batch_matrix_multiply_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/binary.c:147` | named function | setup_binary_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/fully-connected.c:1297` | named function | setup_fully_connected_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/pack-lh.c:178` | named function | setup_pack_lh_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/rope.c:182` | named function | setup_rope_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/softmax.c:131` | named function | setup_softmax_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/static-constant-pad.c:163` | named function | setup_constant_pad_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/static-reduce.c:242` | named function | setup_reduce_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/static-slice.c:151` | named function | setup_slice_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/static-transpose.c:171` | named function | setup_transpose_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/unary.c:251` | named function | setup_convert_operator | xnn_runtime opdata.setup copied from node.setup | preparation |
| XNNPACK `src/subgraph/unary.c:430` | named function | setup_unary_operator | xnn_runtime opdata.setup copied from node.setup | preparation |

## Distinguished: not callbacks, unmarked

- Local helper lambdas and blocks that the declaring function calls directly and never passes on: Metal 202 in all.
  - metal_archive.mm 6, metal_capture.m 7, metal_ir.mm 53, metal_lower.mm 34, metal_native.mm 28, metal_neural.mm 7.
  - metal_program.m 7, metal_recording.m 12, metal_transform.mm 4, metal_transport.mm 22.
  - provider/litert_mesh.cc 21, tools/e2b_replay.m 1.
  - Dawn printer.cc `span`, `pointer`, `direct` and `leading` are likewise direct helpers.
  - A local that is passed by name to another consumer is a callback and is marked: 14 such locals, each listed with
    the lines that pass it.
- Same-named variables a first pass flagged as passed:
  - metal_ir.mm `moved`: its use at line 3709 is a `mesh_graph_access_map *`;
  - metal_lower.mm `point`: line 1077 is a local `ir::Value *`;
  - metal_transport.mm `value`: line 690 is a structured binding.
- Directly recursive `std::function` locals: metal_ir.mm ElementViews `outer` and ByteViews `expand`. Each has
  self-recursion plus one direct call. Their markers were removed after review; both bodies are as committed.
- Direct calls through a local pointer or block:
  - metal_lower.mm `clipped`: a captureless lambda converted to a function pointer, called directly in its own function;
  - tools/e2b_replay.m `span`: a block called directly.
- Ordinary calls through an existing library function address:
  - metal_ir.mm LoweredBody `Two(op, a, b)` applies mesh_graph_index_add, _mul, _floordiv, _python_mod, _minimum,
    _maximum, _and and _or;
  - metal.py `GET`, `PUT` and `_message` are `CFUNCTYPE(...)(SEND)` foreign-call wrappers of `objc_msgSend`;
  - metal.py `_INCREF` and `_DECREF` are `PYFUNCTYPE` wrappers of `Py_IncRef` and `Py_DecRef`;
  - metal.py `RELEASE()` in `_placed_buffer` is a NULL release, with no body.
- Function addresses used for image lookup (`dladdr`), not invocation: provider/litert_mesh.cc `ImageBase` and
  `ProviderPath`, and metal_archive.mm `&metal_ir_archive_save`.
- Swift braces that are statement or declaration blocks, not closures:
  - Bounds.swift:60, an initializer's parameter list;
  - `for`, `if` and `while` blocks at Bounds.swift:103, coreml-chain.swift:93 and :139, gram-chain.swift:118 and
    indexed-gather.swift:67;
  - coreml-chain.swift:12, a computed property, and :49, a function body.
  - swift/Topology.swift has no callback uses and is unchanged.
  - Swift `using: sum`, `using: add` and `using: combine` pass `TensorFunction` values; the closures inside them are
    marked where they are made.

## Cross-language registrations

- **rdma/mesh_graph.py** (owned by 104, not edited):
  - line 25 (`Owned.__init__`) registers `weakref.finalize(self, free, made)`;
  - line 970 registers `LIB.mesh_graph_context_free`;
  - the native bodies `mesh_graph_search_free`, `mesh_graph_closure_free` and `mesh_graph_context_free`
    (rdma/mesh-graph.c) are marked.
- **rdma/mesh_metal4.py:155** (owned by 104) registers `LIB.mesh_metal4_release`; its body in rdma/mesh-metal4.m is marked.
- **rdma/mesh_compile.py:8241** (owned by 104, obsolete path) passes `RELEASE(metal_ir_archive_free)` outward. The body
  (metal_archive.mm) is marked. Only that obsolete path registers it as a callback. metal.py and native callers call
  `metal_ir_archive_free` directly, and the body marker disables those direct uses too.
- **compiler/mesh_compiler** registrations (owned by 68 and 66):
  - `lower_metal.py:244` registers `metal_ir_program_free` as a finalizer; that body (metal_ir.mm) is marked, and is
    also a `std::unique_ptr` deleter.
  - `native.py:1061` registers `MetalRecorderActorsFree` as a finalizer. The active body is metal_capture.m's (built by
    Makefile `RECORDING_SRCS`), and it is marked; metal_recording.m's copy, the inactive alternative, is marked too.
  - `schedule.py:721` registers `mesh_local_free` as a finalizer; its body in rdma/mesh-collective.c is marked.
  - `lower_coreml.py:309`'s finalizer is a Python lambda that calls `mesh_coreml_release` directly, so it is not a
    native callback body.
- **metal.py's own**:
  - `weakref.finalize(..., metal_ir_binary64_free, ...)`: both the registration and the native body (metal_ir.mm) are
    marked.
  - `_RELEASED_OBJECT` is marked at construction.
  - CPython's `Py_DecRef` is external and unedited.

## Uncertain or derived, for root's combined review

- **tools/mesh_signature.h `printers[3]`** is a static table of `mesh_graph_matrix_market`, `mesh_graph_dot` and
  `mesh_graph_space_time`, which the tool indexes and calls. That is dispatch through library addresses within one
  component, not a registration with another, so it is unmarked.
- **metal_ir.mm LoweredBody `argumentAt` and `valueAt`** are `std::function` members invoked by LoweredBody's methods.
  The constructor's default bodies and RaisedModule's replacement bodies are all marked as stored callbacks; the defaults
  are the same component's own.
- **Dawn intrinsic tables:** the four TypeMatcher match/print pairs on Mesh-added lines are marked, in
  `src/tint/lang/core/intrinsic/data.cc` (6 markers) and `src/tint/lang/msl/intrinsic/data.cc` (2). Tint's intrinsic
  generator emits those files from `.def` definitions. The Mesh-added `.def` entries they derive from were not edited,
  and no generator ran. Unrelated table bodies are untouched.
- **Python callbacks from other owners**, crossing into native APIs; their Python bodies are not in this grant:
  - rdma/mesh_graph.py:709 `ROW_KEY` and :994 `_freed`;
  - rdma/mesh.py:309 `RELEASE` and :353 `EXTENT`;
  - rdma/mesh_metal4.py:70 `HOSTED`;
  - rdma/mesh_compile.py:1874 `DELETER`, :8677 `_RELEASED_OBJECT` and :8852 `_dropped`;
  - compiler/mesh_compiler/bridge.py:57 `_retired` and native.py:56 `Owned.released`.
  - Their native consumers are marked only where a Mesh native body is what gets passed.
- **provider/litert.patch** was outside the original131 grant. The explicit Codex follow-through below closes that maintained-patch coverage gap; the original uninspected status is superseded.

## Excluded

- Vendored dependency trees: xonotic, third_party, and upstream Dawn and Tint bodies outside the Mesh-added patch lines.
- Generated output_data and build outputs.
- rdma and other non-compiler Python (104's), and compiler files other than metal.py (68's and 66's).
- Mesh bench/ and user/ and metal-microbench tools/working_set.m were scanned and contain no callback sites.

## Codex follow-through: maintained provider patches

The operator assigned this remaining slice directly to Codex after the initial sweep. Scope is Mesh-owned additions,
changes and moved callback uses in maintained provider patches; it is not every callback in the upstream dependencies.
`git ls-files provider` contains exactly two maintained patch files: `dawn.patch` (the initial30-site scope) and
`litert.patch`, which embeds `xnnpack_mesh.patch` and `ml_drift_mesh.patch`. All three LiteRT patch layers were inspected.
No additional maintained provider patch was found. Provider build files and original vendored callback machinery were
not rewritten. No Claude process received a continuation.

Every new marker is a native **compile-time error mechanism**: it takes the address of the undeclared
`mesh_callbacks_are_outside_spec`. The phase in the tables is the original callback's use, not a claim that the disabled
callback was executed. In particular, eight compilation/construction callbacks are marked without an exemption for their
phase. The53 sites comprise41 named callback bodies, six supplied-object virtual methods and six forwarding/handoff
sites;28 preparation,16 execution,8 compilation/construction and1 release.

### Actual caller evidence

- LiteRT's `TfLiteRegistration` entries in comparisons.cc:492–529, logical.cc:168–179, reshape.cc:252 and
  select.cc:216–230 register the marked Init/Free/Prepare/Eval bodies. The two Select registrations share one template
  definition, which receives one authored mark. The modified WebGPU `Invoke` and `DelegatePrepare` are bound by
  kRegistration.invoke and TfLiteDelegate.Prepare. Direct Select/PrepareShape/Data helpers are not registrations.
- `LogicalOr` and `LogicalAnd` are supplied to original `reference_ops::BinaryFunction` and
  `BroadcastBinaryFunction4DSlow`; internal/reference/binary_function.h invokes `func` on each value. The two
  `ComparisonString` handoffs supply original StringRef predicates to the corresponding reference algorithms.
  Their unmodified upstream predicate bodies remain untouched.
- XNNPACK's twelve modified subgraph setup bodies are assigned to `node->setup`. Original runtime.c:741 retains that
  address as `runtime->opdata[i].setup`; runtime.c:373 and:1047 invoke it. `setup_convolution_operator` is a direct helper
  called by the original registered `setup_convolution_operator_static`, so it was not misclassified by its name.
- The new `XNN_PARALLELIZE` macro's `selected == NULL` branch forwards the task operand to pthreadpool. One authored mark
  precedes that handoff and expands in all38 actual parallelization cases. The non-null selection branch retains
  function/task/operand identities as data and is not itself a callback invocation. Its source metadata was preserved.
- Mesh supplies OpSelector/IrModelOpSelector objects at the two new DelegateKernel GraphToGpuModel/IrModelToGpuModel
  handoffs. Original MLDrift ConvertOperations calls their GPUSubgraphFromGraph/GPUOperationFromNode methods
  (gpu_model_util.cc:352/376, ir_model_util.cc:393/429). Markers sit at the new handoffs, without editing those original
  selector implementations.
- Mesh moved `DescriptorSpatialTensor` into merge_nodes.h and supplies it through GPUOperation::SetSrc/SetDst
  (merge_nodes.cc:535/542 and DelegateKernel::MeshEmit). Original RecalculateGridSize uses Width/Height/Depth/Slices/Batch
  (gpu_operation.cc:1531–1549); original kernels such as conversion.h:75 and strided_slice.cc:207 use Channels. Those six
  supplied-object methods are marked. GetDescriptor has no demonstrated callback use for these supplied descriptor
  instances, and the defaulted destructor is not treated as a callback merely because it is virtual; both remain intact.
- WebGpuArguments::Init's moved UniformBufferCreator call is an actual caller-supplied interface invocation. Only that
  moved call is marked; the separate unchanged Init overload's identical upstream machinery is left untouched.

### Distinctions and source limits

`mesh::Site`'s main selected numerical function applies an existing library function address directly to its operands;
its retained SelectedBody/Function identities and shape data are not callback registrations. Logical/string predicates
passed onward into reference algorithms are callbacks and are marked above. XNN config initialization records,
relocations, task_name strings and default-visibility declarations represent selection/binding facts, not newly authored
callback bodies. The38-case handoff covers their actual pthreadpool use without marking every unchanged xnn_compute_*
body. Existing variant_setup_fn machinery in batch-matrix-multiply was not modified by Mesh and was not edited wholesale.

The MLDrift origin/resource/dispatch/converter additions otherwise make direct calls or retain data. Ordinary std::sort
without a supplied comparator is not a newly authored callback. Full dynamic targets of arbitrary caller-provided
interfaces are not established by this source review; the actual Mesh-owned handoffs and bodies are identified above.
No source-only count is a runtime coverage claim.

### Preservation and patch checks

- Original XNNPACK `55a0ae4` → `877af3f3f422e865177e7c038133c160895aa34d`:13 marks in12 files.
- Original MLDrift `576d069` → `239994d65028394a283f0f28f89eb2f8c918070f`:7 marks in2 files.
- Original LiteRT `767121aa` → `fbde6d03516ccbf63ef8c13e789f86daae131958`:33 marks in8 files, plus regenerated nested
  patches from the two committed source results. No uncommitted source was transferred between checkouts.
- Metal `1703f8df960d0ef415b9c8d3dde075e136c5f650` preserves only the regenerated provider/litert.patch.
- Removing only marker/comment insertions restores every one of the22 source files byte for byte. All three dependency
  source diffs pass `git diff --check`. Outer patch-text checking reports three blank unified-diff context lines as
  trailing whitespace; each is exactly one required context-prefix space, not source whitespace introduced by a marker.
- Each maintained patch exactly equals its committed source diff. Reverse `git apply --check` passes; forward application
  from the original base into a private temporary index reproduces the exact committed tree for all three patch layers.
  No checkout or source backup was used for this check, and no compiler or dependency generator ran.
- The canonical `.build/provider/litert` checkout was clean at historical `f9eac343`, whose nested XNN patch predated the
  maintained source. Git fetch/check-out now realizes the complete committed `fbde6d03` source; its tracked tree is clean
  and litert.applied contains the identical committed patch. The prior realized source is retained by the annotated
  `native139-realized-source` tag. Bazel external cache trees were not edited; build stamps and binaries were not changed.

Patch SHA256 values: xnnpack_mesh.patch `441aeff59c6b64d163b3f1b2c635519e12ad419ff1bb3e788e4284c33f5818cb`;
ml_drift_mesh.patch `a13fb71a0aa1f9acac8373d11d5a84cf05549bae8abfd9fcc3eed7008956e498`;
provider/litert.patch `732dd38fe0b5a89d4fd2c7a6672e9ff11ef25fb38db663b944ac640a75a87b8f`.

These are source preservation/projection results only. No import, native build, callback execution or workload ran.
The intentionally disabled source is not represented as matching historical native140 binaries. General producer
integration remains paused.

## Process

- Detection and editing were static only. Scripts in /tmp/sweep read the sources, inserted the markers, and checked that
  every body was retained; no source copies were made. Each classification was rechecked against the surrounding source.
- `python3 -m py_compile compiler/mesh_compiler/metal.py` ran once, at 23:06:47, after the first metal.py markers. It is a
  source-only bytecode/syntax check, authorized as source-only verification. It establishes neither callback execution
  nor native build success, and was not repeated; every later check is text or AST only. No Mesh or Torch import,
  native build, callback execution or workload ran.
- Preservation after source HOLD: canonical Dawn `09457b098` and Metal `2c634181`; the Mesh native/Swift/metal.py sources and this inventory are included in the accompanying scoped Mesh checkpoint. No build or execution followed these commits.
