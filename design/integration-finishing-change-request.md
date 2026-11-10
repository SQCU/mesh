# Change request: finishing the torch lowering and capture integration

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
  - The mini's `configuration.plist` is corrected the same way.
  - Start every bridge with `bin/mesh-bridge.sh`.
  - Run e2b on all five machines to its end: every rank's prefill and decode end, and the run's record states each
    rank's decoded tokens. Where a rank stops, the code or configuration that stops it is diagnosed and corrected (the
    operator, above: "finish the integratino work here"; "it's the job of hte agent to diagnose and correct the missing
    or wrong configuration").

### 2. A node's replayed commands in the completion-time order

- **The algorithm** is list scheduling in the order of the linear program's completion times (Hall, Schulz, Shmoys and
  Wein; design/list-scheduling-sources-2026-11-08.md, LS).
- **It takes** a key for each of a node's replayed commands, and the commands' dependences. A command that reads a
  received range has that range's published completion time as its key (`struct mesh_state` `times`). Every other
  command has its dependence level as its key.
- **It writes** the node's list of commands: `mesh_graph_heap_order` over those keys.
- **Consumed by** the node's replay, which executes its commands in that list's order: Lee and Ha's self-timed order,
  as design/torch-source-lowering-change-request.md item 3 quotes it.

### 3. A weight on each task

- **The data structure** is the weighting operand, of the type design/aot-compilation-change-request.md A3 gives: "a
  weight on each task; a weight on each edge; the cables' capacities, from the map". A3's task is a function. The
  functions' weights are the lengths of `mesh_plan`'s functions.
- **Produced by** the caller of the compilation. **Consumed by** the compilation's router and its scheduler, which
  compute no weight (A3).
- **Replaces:** the weighting operand of one `mesh.Weight` per transfer.

### 4. The masked pass in the graph module

- **Replaces:** the walk in metal-microbench metal_recording.m `MetalMaskedPass`.
- **Write:** that walk is a function of mesh rdma/mesh-graph.{h,c}, with caller operands.

### 5. The withdrawn item's code

- **Replaces:** metal-microbench c4d41929's `dense_tensor_f16_received` and tests/gpu/received_matmul.m.
- **Source:** design/torch-source-lowering-change-request.md item 4, withdrawn 2026-11-08.
- **Write:** both are deleted.

### 7. Each commit's feedback

- **Source:** design/commit-feedback-sources-2026-11-08.md (CF) §1: `addFeedbackHandler:` "Registers a commit feedback
  handler that Metal calls with feedback data when available." The headers do not say how many commits one handler
  serves. CF §3 measures it: "With one options instance reused, eight commits called its handler once. With one options
  instance a commit, each of the eight called its handler." MTL4CommandQueue.h L261 signals an event's value "after all
  GPU work prior to this point is complete".
- **The data structure** is one `MTL4CommitOptions` a commit, whose feedback handler was added for that commit alone,
  and that commit's record: struct Feedback's semaphore, error and GPU times, and struct MetalRun's `failure`.
- **Produced by** every commit of the replay: the handler writes the feedback data Metal calls it with to its own
  commit's record. **Consumed by** `feedback_wait`, which receives one handler's call a commit.
- **Replaces:**
  - in metal-microbench tools/e2b_replay.m, `feedback_start`: one `MTL4CommitOptions` a phase (`PREPARE`, `PREFILL`,
    `DECODE`, `PROFILE`), which every commit of that phase carries;
  - in metal_recording.m, `MetalRunOf`: one `MTL4CommitOptions` that every replay's commits carry.
