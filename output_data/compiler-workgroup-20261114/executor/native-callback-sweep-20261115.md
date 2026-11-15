Mesh’s required architecture represents allocation, lifetimes, placement, dependencies, and schedules as explicit data because these are solved ahead of execution and must remain available to later transformations.
Producers publish complete results in paged storage, and consumers access those same results by view, preserving their identity and page-table references.
Compilation therefore composes functions of explicit operands and represented effects, with original objects or argument bundles passed through existing interfaces using `*args` and `**args` where appropriate.
This is a stronger functional discipline than ordinary Python programming: every agent must recognize copying or reconstructing existing results as a departure from the specification and correct its work.

Every descendant function accesses produced structures through views of the same storage, with their page-table references carried intact.
Passing a view preserves access to the complete structure while each consumer dereferences exactly what its algorithm uses.
Packing, unpacking, or selecting fields into smaller replacement structs along that chain is unnecessary because the complete structure is never passed by copy.
Throughout this integration, agents use access by view to avoid spending memory bandwidth and stack work on duplicated representations, especially across Apple Silicon’s bandwidth-limited interconnect.

# Native callback sweep — operator priority, 15 November 2026

Released to the existing SAME131 session e48f4bc7-7cef-4afd-ae50-fc1b4618fa73 / PID78966 / exec50810. Its producer turn is interrupted (control native131-callback-sweep-priority success, no queued input, result74 aborted_streaming). Producer WIP is preserved at Metal1dca29d4 and Mesh24481f5f. This priority temporarily replaces the producer implementation; resume that original task only after this sweep is reported and root directs it.

The user explicitly requires a quick sweep of ALL callbacks used in Mesh: retain their bodies, deliberately induce a compile/runtime error at each actual detected callback, and add a comment explaining that callback inversion of control is outside the working specification and runtime callback overhead is inadmissible in performance-critical code. Record the callback's actual phase (compilation/construction, preparation, execution, completion, or other actual phase). Compiler-construction callbacks, including the Protocol builder callbacks, are included.

## Disjoint source ownership

- Mesh-owned C/C++/Objective-C and headers under /Users/mdot/dox/mesh/rdma and other Mesh native paths outside compiler. All rdma Python belongs to104; do not edit it.
- Mesh-owned Swift source and examples, including swift/Bounds.swift, swift/Topology.swift and examples/*.swift. Include original DispatchQueue.main.async callbacks such as indexed-gather/coreml-chain.
- Mesh-serving native source in /Users/mdot/metal-microbench: the eight METAL_IR_SRCS and their headers; metal_capture, metal_program, metal_recording, metal_submission; provider/litert_mesh native code; Mesh tools including tools/mesh and tools/e2b_replay.m, and Mesh native clients where callbacks occur. Exclude unrelated serving/UI implementation.
- compiler/mesh_compiler/metal.py ONLY within the compiler Python package.68/66 own other compiler Python/patches;104 owns export/ir and Python outside compiler.
- provider/dawn.patch only for Mesh-added callback sites. Do not modify original vendored callback internals. Inspect the authored diff/source ownership before marking a dependency callback.
- Exclude vendored xonotic/dependency trees, generated output_data and compiled/generated outputs.

## Markers and retained source

At native callback bodies use a genuinely undeclared function address in a local typed pointer binding, e.g. `void (*mesh_callback_disabled)(void) = &mesh_callbacks_are_outside_spec;`, with the requested explanatory comment. A string-to-function-pointer conversion may only warn in C and is insufficient. The undeclared identifier produces a compile diagnostic without an unsafe dereference or undefined runtime behavior. Preserve the original body after it. Swift callback closures use an unresolved callback marker so compilation fails. Python callback bodies use an unresolved callback marker so invocation fails; preserve the expression/body and place the explanatory comment at the detection.

Distinguish callbacks from ordinary direct calls through an existing library function address. Include registered function-pointer bodies, std::function/algorithm/IR-builder callbacks and Objective-C/Dispatch/Metal completion blocks. For indirect targets whose actual owned body cannot be established, retain the source and record the exact registration/caller/target uncertainty for root's combined review; do not guess that every ordinary function is a callback. No alternate dispatcher, fallback or body deletion.

Produce an inventory at output_data/compiler-workgroup-20261114/executor/native-callback-inventory-20261115.md: file, symbol/site, callback consumer, actual phase, marker/comment and any uncertain indirect target. Keep bodies/files in place and summarize marked sites by file/phase. No build, compiler invocation, import, workload or runtime execution during deliberate disablement. Source reads/edits and static lexical/AST inspection are allowed. Do not create new workers or restart any existing process. Return a stable source boundary for Codex inspection and ordinary named commits.


## Completion receipt

SAME131 terminal success/source HOLD at 2026-11-15T23:15:33Z. 645 marks are inventoried in native-callback-inventory-20261115.md. All original bodies remain; active recorder ownership is established from Makefile RECORDING_SRCS and its inactive alternative is distinguished. Verification is text/AST/diff and the one source-only metal.py bytecode/syntax check, not native build or callback execution. The original producer remains paused; 132/133/135 remain stopped.

Preservation: canonical Dawn `09457b098`, Metal `2c634181`, and the accompanying named Mesh source/inventory checkpoint.
