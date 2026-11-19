# Mesh graph transformations and final lowering, 18 November 2026

## A: What the removed entry did

Mesh’s driver accepted a Python module, example tensors and optional submodule names. It tried to capture the program, discover a bridge, choose a seed, lower numerical bodies, install a target and return an executable inside a schedule. The capture producer and the driver already disagreed about the input record. The tensor-parallel and engine examples also asked that entry for a finished program. The cutover removes the driver and those launch paths. The surviving example modules contain numerical workloads and placement policies. The older [entry review](captured-entry-review-2026-11-18.md) records the prior behavior.

## B: What goes into and comes out of Mesh

The application gives Mesh a call graph. Each call names its callee, its operands and where its results go. Mesh returns a transformed graph. A producer feeding a consumer imposes a dependency on the relevant operand or page. Disjoint accesses can permit either order when the remaining dependencies and effects permit it. The solver can weigh those orders together with sharding, routing, fusion and allocation. A target can express the selected ordering through queues, instructions, or device commands. The application still calls the program with its own inputs. [AOT A1–A10](aot-compilation-change-request.md) · [Joint selection](compilation-composition-change-request.md)

The compiler writes its executable and IR results into the caller’s DMA-accessible page extents. The caller can pass a live view to another compilation or free the pages after the last reader finishes. This permits reuse across parameter changes without obligating the caller to keep a previous executable. [Functional compilation](functional-compilation-memorandum.md) · [Compile, invoke and retire](compiler-pseudocode.md)

### What the last wave changed

The source work retained FX operation identities and found legal communication regions from their dependencies. The joint-selection work kept alternatives and their costs so that placement, routing and fusion could influence the same choice. The storage and recorder work carried the selected order, lifetimes and page bindings into the target. Evaluations then exercised numerical output, aliases, selected alternatives and Metal replay. Finally the whole-program check caught a distinct-callee error that separate body runs had concealed. This is the flow of operands between scopes. Joint selection can evaluate the free choices in any order that preserves their constraints. [Revisions and runs across the five repositories](../../../compiler-emit/artifacts/joint-selection-review-20261118/implementation.md)

### Outstanding properties

The [19 November revision queue](../output_data/compiler-review-20261119/urgent-revisions.md) identifies the cross-call placement change and traces which compiler properties the placement, fusion, replay and gradient evaluations actually exercised.

| Scope | What the reviewed path establishes | What still has to hold |
|---|---|---|
| Frontend, complete call graph | The multi-function producer reads the outer FX graph. Its call nodes name the bodies and bind their arguments. The producer projects FX dependencies and storage accesses into the planner. | Check the same projection on joint forward/backward graphs and the identity and advancement of random generators across calls. [Source constructor](../../../compiler-emit/compiler_emit/source.py) · [A1 and A10](aot-compilation-change-request.md) |
| Joint optimization | The finite-choice run exercised four interacting choices and retained selections with different costs. | The full program must retain its forced edges separately from its free choices and weights, including independent-page readiness, contiguity, routes, partitions, fusion and memory reuse. Every lowerer must receive the same selected graph, with the solver results that justify its order and offsets. Coverage remains open across the admitted program domain. [Joint-selection exercise](../../../compiler-emit/artifacts/joint-selection-review-20261118/evaluate-forms.py) · [Joint IR](../../../compiler-emit/docs/joint-ir.md) |
| Storage across executions | The native plan writer can write a physical compilation directly into a supplied Mesh section and reopen its arrays by view. Metal installation also has a file-backed recording path. | Write the complete callable and its bindings to caller-owned extents. A caller that keeps an IR extent can pass its view to another compilation. The physical-plan writer establishes residency for the plan alone. [Section writer](../../../native-execution/native_execution/planning.py) · [Metal installation](../../../compiler-emit/compiler_emit/metal_install.py) |
| Final lowering, complete program | The CPU package and the local Mesh Metal recording execute the outer FX graph. The Metal producer compiles each typed callee once, and the recorder reads its function index at each occurrence. Fresh and reloaded Metal recordings matched eager FX for the repeated callee, an input mutation and a strided intermediate. | Multi-processor emission must consume the channels in the selected plan. The local comparisons exercise numerical composition, dependencies and storage views. Whole-program multi-rank execution and its cost remain open. [Whole-program check](../../../compiler-emit/artifacts/joint-selection-review-20261118/evaluate-calls.py) · [Recorder](../../../compiler-emit/compiler_emit/metal_install.py) |
| Final lowering, overlap and cost | Specific source cuts, aliases and saved Metal recordings have run. The two-worker reference runs the same placed program with FX bodies and with compiled CPU bodies. The reviewed built-in target table names Torch, MPS AOTI and Mesh Metal. | For MPS, ANE, CPU and mixed placements, actual execution must implement the selected page dependencies, offsets and transfers. An independent call may proceed while an unrelated page is late. A dependent call waits for its own required data. Each target must bind parameters, initialize its constants, and lower every numerical and transport operation assigned to it. Whole-program ANE and mixed-engine lowering remain open in this path. Complete multi-body, multi-rank execution and its launch and synchronization costs still need measurement on the selected program. [Target table](../../../compiler-emit/compiler_emit/targets.py) · [Lowerer contract](lowerer-interface-memorandum.md) · [Execution observations](../../../compiler-emit/artifacts/joint-selection-review-20261118/implementation.md) |

### A: How a call disappeared

The evaluation gave the planner function indices, edges and buffer bindings by hand. It spelled out the same call order again in the Python expression for the expected answer. The planner received no outer FX graph. Target construction introduced a function selector and gave it the default value zero. The planner scheduled both bodies, but target construction indexed the schedule with zero and sent only the first body to the emitter. The package correctly executed the one body it received. The original numerical discrepancy was 1.31. The term “CPU reproducer” referred to the evaluation that exposed this mistake. [Original observation](../../../compiler-emit/artifacts/joint-selection-review-20261118/implementation.md)

### B: Who supplies each operand

The caller supplies the outer FX graph once. FX supplies argument and result references at each call. The compiler derives the integer function indices required by the native planner and binds the selected bodies at those same FX nodes. The CPU evaluation calls the original graph eagerly for its expected answer, then loads the emitted package and calls it on the same inputs. A second graph calls the same function on two different extents. Between those calls it writes to an input that the last call reads. The check compares the returned values and the write to the input. The compiler derives both call sequences from FX. [Planner and evaluation](../../../compiler-emit/artifacts/joint-selection-review-20261118/evaluate-calls.py) · [Numerical target](../../../compiler-emit/compiler_emit/targets.py)

### A: What the Metal producer and installer did

The planner wrote the callee index into `planned.call[i].function`. The Metal target accepted one numerical function. The installer took its handle and repeated it once per call. The public Metal entry stopped at target construction for an outer graph with distinct callees. The repeated handle came from the installer’s code. [Target construction](../../../compiler-emit/compiler_emit/targets.py) · [Implementation review](../../../compiler-emit/artifacts/joint-selection-review-20261118/implementation.md)

### B: How the emitter reads the plan

The application gives the compiler the graph once. The Metal allocation associates each numerical program it produces with its function index. For call `i`, the emitter reads `planned.call[i].function`, looks up that program, and passes its call-specific placements to the shared Metal recorder. For an outer graph calling `A, B, A`, Mesh compiles the two bodies in one graph-analysis context and gives both to one allocation. The three recorded calls read the plan’s three storage bindings. The mutation evaluation also caught a Metal writer defect: it used the backing buffer name for two different kernel parameters. The writer now associates each read or write with the formal parameter used in that instruction. [Metal allocation](../../../compiler-emit/compiler_emit/storage.py) · [Metal emitter](../../../compiler-emit/compiler_emit/metal_install.py) · [Shared recorder](../compiler/mesh_compiler/native.py)

### Reproduce the placement check

From the Mesh checkout, with its three sibling compiler projects available:

```sh
PYTHONPATH="compiler:rdma:$HOME/schedule-emit:$HOME/compiler-emit:$HOME/native-execution" \
  "$HOME/compiler-emit/.venv/bin/python" output_data/callgraph-20261118/run.py
```

The [check](../output_data/callgraph-20261118/run.py) places the outer FX graph `A, B, A` and checks its numerical results, gradients, output aliases and observable state. A second program shards a matrix over two CPU processors with unequal widths. Both the reference workers and workers calling the compiled CPU bodies run that same scheduling result on three inputs. These are local observations of the graph interface.
