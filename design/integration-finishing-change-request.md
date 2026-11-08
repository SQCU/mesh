# Change request: finishing the torch lowering and capture integration

Code is cited at mesh 9b1cc225 and metal-microbench 0a910421.

## The turns

> nothing is 'still open'. if there's no clear explanation of how to finish an integration work segment from the cited literature, there was no cited literature int he directions givcen to the implementation agents, which is the fault of the change request for underspecification or you for not passing the change request along. finish the integratino work here.
> "  Starting e2b needs your call on enabling the job, the Ultras' region and the block size."
> there is no 'call' here. if they are misconifgured but responsive, it's the job of hte agent to diagnose and correct the missing or wrong configuration.

(2026-11-08)

## Requirements

### 1. Every mesh machine's bridge runs

- **Source:** the operator, above: "if they are misconifgured but responsive, it's the job of hte agent to diagnose and
  correct the missing or wrong configuration."
- **Write:**
  - Diagnose each machine's bridge configuration against the configuration the repository installs (mesh
    bootstrap.sh, install.sh, bin/mesh-bridge.sh and the files they write): the `io.mesh.bridge` job, `bridge.conf`'s
    region, and the block size. Correct each one so that all five machines agree.
  - The mini's `configuration.plist`, rewritten by a `mesh-bridge.sh start` on 2026-11-08, is corrected the same way.
  - Start every bridge with `bin/mesh-bridge.sh`. Never SIGKILL a bridge.
  - Run e2b on all five machines, never a pair or a subset.

### 2. A node's replayed commands in the completion-time order

- **Source:** design/list-scheduling-sources-2026-11-08.md (LS): Hall, Schulz, Shmoys and Wein, list scheduling in
  the order of the linear program's completion times. Lee and Ha's self-timed order, as design/torch-source-lowering-change-request.md
  item 3 quotes it.
- **Write:** a replayed command that reads a received range is keyed by that range's published completion time
  (`struct mesh_state` `times`, mesh a582ce3c). Every other command keeps its dependence level. The list is
  `mesh_graph_heap_order` over those keys.

### 3. A weight on each task

- **Replaces:** mesh 9b1cc225, which narrowed the weighting operand back to one `mesh.Weight` per transfer.
- **Source:** design/aot-compilation-change-request.md A3, quoting its type: "a weight on each task; a weight on each
  edge; the cables' capacities, from the map".
- **Write:** the weighting operand gives a weight on each task and on each edge, as 14b2061d had it. The task weights
  are the lengths of `mesh_plan`'s tasks.

### 4. The masked pass in the graph module

- **Replaces:** the walk in metal-microbench metal_recording.m `MetalMaskedPass` that C9 flags at 5600d84b.
- **Write:** that walk is a function of mesh rdma/mesh-graph.{h,c}, with caller operands, and C9 passes.

### 5. The withdrawn item's code

- **Replaces:** metal-microbench c4d41929's `dense_tensor_f16_received` and tests/gpu/received_matmul.m.
- **Source:** design/torch-source-lowering-change-request.md item 4, withdrawn 2026-11-08.
- **Write:** both are deleted.

### 6. The index of change requests

- **Write:** design/change-requests-index.md states each request's state after these commits.
