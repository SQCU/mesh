# The compiler and the runtime: stages, representations, and the interfaces between modules

The quotations are from these records:
- design/scaling-models-sources-2026-11-07.md (SM);
- design/modes-and-strategies-sources-2026-11-07.md (MS);
- design/process-network-sources-2026-11-07.md (PN);
- design/specialization-sources-2026-11-07.md (SP);
- design/joint-optimization-sources-2026-11-07.md (J);
- design/completion-and-period-sources-2026-11-07.md (CP).

Code is cited at mesh d980d4d4 and metal-microbench bed56a95 as file:line.

## 1. The machine model

- **Devices.** The repositories support every Apple silicon device with RDMA over Thunderbolt. TN3205 gives the
  transport: "Send and receive operations only" (MS §1).
- **Dies.** A device has several dies on one package. The operator, 2026-11-07:
  - M5 Max is 1×2 dies and M5 Ultra 2×2;
  - at least four M5 Ultra devices are in the collective;
  - "it is strictly more true that there are high diecount high interconnect nodes than single nodes".
- **Paths.** A transfer takes one of three paths:
  - die to die on the package;
  - memory to the fabric;
  - Thunderbolt between devices.
- **Cost of a transfer.** Each path costs "α + nβ, where α is the latency (or startup time) per message, independent of
  message size" (SM §3), with its own α and β.
  - For small n, α dominates.
  - The operator: a small write across machines takes about as long as a memory-to-memory write within one machine.
- **The model the compiler reads.** A graph of devices and links, each link with its (α, β). There is no count of nodes
  in it.
- **Memory.**
  - Each device registers one region with the transport once (data-model.md §2).
  - The region is divided into pages; "a page is a region of memory" (operator).
  - A page table names each page (data-model.md §6).

## 2. The compiler

A compiler is required. TN3205 has no remote write, read or atomic, so "the receiver fixes where each frame lands, and
in what order, before it comes" (data-model.md §3). Every address and every order is therefore decided before the
call.

The compiler is three front ends, a middle that solves one joint problem (design/joint-compilation.md), and back ends.
Each stage below gives:
- the representation it reads;
- the representation it writes;
- the method;
- the module.

### 2.1 Front ends: the representations the compiler reads

| Front end | Representation read | Its definition | Module |
|---|---|---|---|
| PyTorch programs | `ExportedProgram`: an FX graph of Core ATen operators, an `ExportGraphSignature`, a `state_dict` | Core ATen "is fully functional, and there is no `inplace` or `_out` variants" (PN §4). The signature's `BUFFER` inputs and `BUFFER_MUTATION` outputs are the values carried between calls (PN §4) | mesh_compile.py `program` (:2122), `_Export`, `IR` (:53) |
| LiteRT models | a TFLite flatbuffer: `Model`, `SubGraph`, `Operator`, `Tensor`, `Buffer`, `SignatureDef` | metal-microbench `.build/LiteRT` tflite/converter/schema/schema.fbs:293, :1615, :1665, :1690, :1736, :1753 | tools/prepare_e2b.py `prepare`, which writes the manifest of stages, routes and kinds |
| Metal command streams of another runtime | the commands one call issues, recorded with their operands | a residual program: "running r on d2 yields the same result as running p on both d1 and d2" (SP §1, Definition 4.1) | metal_recording.h `MetalRecorderCapture` (:76) |

### 2.2 The middle: one joint problem

| Stage | Reads | Writes | Method | Module |
|---|---|---|---|---|
| Sharding | the IR | a sharding per operator | GSPMD's propagation (J §5) | mesh_compile.py `partition` (:610) |
| Collective lowering | the sharded IR | collectives as declared transfers (`struct mesh_declared`, mesh-collective.h:36) | ring and tree algorithms (Patarasuk & Yuan; Rabenseifner; Thakur, Rabenseifner & Gropp), chosen by α + nβ (SM §3) | mesh_compile.py `lower` (:1091), `unit` (:1140) |
| Collective schedule and placement | `struct mesh_function`: declared transfers, weights, dependences (mesh-collective.h:24–50) | the published operand: `struct mesh_algo`, `struct mesh_state`, `struct mesh_hop` (mesh.h:80–144) | design/joint-compilation.md §3, rows 3–8. The linear program is solved by HiGHS, "a high performance serial and parallel solver" (SM §4) | `mesh_plan` (mesh-collective.h:64) |
| Shares across calls | each call's segment times | the next call's configuration | design/joint-compilation.md §3, row 9 | rdma/allocate.py |

### 2.3 Back ends: the representations the runtime reads

| Back end | Writes | Module |
|---|---|---|
| Transfers | `mesh_plan`'s published operand, read by the bridge | `mesh_plan` |
| Kernels, specialized | a residual program as Metal 4 indirect command buffers, `struct MetalReplay` (metal_recording.h:8) | metal_recording.h `MetalRecorderReplay` (:77) |
| Kernels, generated | none yet. Inductor's Metal codegen is registered with `register_backend_for_device`, whose arguments include a scheduling and a wrapper codegen (PN §4) | not written |

## 3. The runtime

The runtime is a dataflow process network: "A sequence of such firings is a particular type of Kahn process that we call
a dataflow process. A network of such processes is called a dataflow process network" (PN §1).
- **The synchronous, complete, contiguous consumer** is a functional firing.
- **The asynchronous, incomplete, noncontiguous consumer** is an actor with state.
- **Values carried between executions** are delays with initial tokens (PN §1).

Its parts and their interfaces:

| Part | Interface | Semantics (cited) | Module |
|---|---|---|---|
| The bridge, one process per device | reads `struct mesh_algo`, `struct mesh_state`, `struct mesh_hop` | posts each receive before its send: ready mode, "may be started only if the matching receive is already started" (MS §2); sends in the compile-time order: Sriram's ordered transactions (MS §4); waits only for data: the self-timed strategy (MS §3) | mesh-flow.c and the other bridge files |
| Starting execution t | `mesh_transfers_start` (mesh-call.h:95) | persistent requests, "Create (Start Complete)∗ Free" (CP §1) | rdma/mesh-dataflow.c:263 |
| The GPU's waits and releases | `struct mesh_metal_wait`, `struct mesh_metal_publication` (mesh-metal.h:9–14) | one release store after the writes (design/torch-capture-change-request.md item 5) | rdma/mesh-metal.m |
| The host's publish and arrival | `mesh_host_publish`, `mesh_host_arrived` (mesh-collective.h:68–69) | as the GPU's | rdma/mesh-collective.c:15, :26 |
| Issuing execution t's commands | `MetalRecorderRun` (metal_recording.h:82): execution t starts once t − lag has ended | BSL (1), start(v_i, k) ≥ end(v_j, k − delay) (CP §4) | metal_recording.m |

## 4. Which interfaces are connected

| Producer → data structure → consumer | Connected? | Where |
|---|---|---|
| `ExportedProgram` → `IR` → sharding → collective lowering → `struct mesh_function` → `mesh_plan` | yes | mesh_compile.py `program`: `_Export`, `partition`, `lower`, `mesh_plan` (:2201) |
| PyTorch operators → kernels | through capture only: torch's eager MPS kernels run inside `MetalRecorderCapture`. No lowering of Core ATen to generated kernels | mesh_compile.py `_captured`, `_steps` |
| TFLite flatbuffer → manifest → `struct mesh_function` → `mesh_plan`; recorded commands → `struct MetalReplay` | yes, for E2B | tools/e2b_replay.m: `capture:` (:713), `mesh_plan` (:786), `replay:` (:869), `mesh_transfers_start` (:1107) |
| The capture's residual program → every later execution | for the first execution's static store only. Program point specialization makes the reachable static stores the residual program's points of control (SP §1, Definition 4.4) | design/specialization-sources-2026-11-07.md |
| Declared asynchronous, incomplete, noncontiguous consumer → its consumption, one partition at a time | declared (`MESH_PAGES`), not consumed per partition | design/torch-capture-change-request.md item 3, a stop |
| Completion; offsets for contiguous operands; final lowering; every node's replay | not written | design/torch-capture-change-request.md items 4, 6, 8, 9, stops |
| Sharding ↔ collective schedule ↔ placement | in sequence, not joint | design/joint-compilation.md §4 |
| The Swift engine's forward graph → the compiler | not connected. The earlier TP2 sources (meshtp) were deleted in metal-microbench 0c7466e8; their build products remain in `.build` | metal-microbench |

## 5. Documents this one supersedes in part

- design/data-model.md §1 describes one device class. §1 above gives the machine model.
- design/joint-compilation.md is the middle's optimizers, cited from §2.2.
