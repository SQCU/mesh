# Change request: finishing the torch lowering and capture integration

Code is cited at mesh 9b1cc225 and metal-microbench 0a910421; item 7's at mesh 5d000f5e and metal-microbench 03298a79.

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
  - Run e2b on all five machines, never a pair or a subset, to its end: every rank's prefill and decode end, and the
    run's record states each rank's decoded tokens. Where a rank stops, the code or configuration that stops it is
    diagnosed and corrected (the operator, above: "finish the integratino work here"; "it's the job of hte agent to
    diagnose and correct the missing or wrong configuration"). Item 7 corrects the stop of 2026-11-08.

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

### 7. Each commit's feedback

- **Replaces:**
  - In metal-microbench tools/e2b_replay.m, `feedback_start` (:102–115): one `MTL4CommitOptions` a phase (`PREPARE`,
    `PREFILL`, `DECODE`, `PROFILE`), which every commit of that phase carries (`commit_bounded` :128; the prefill's
    chunks :1081; the decode's steps :1144, and `MetalReplayEncode` :1149).
  - In metal_recording.m, `MetalRunOf` (:4476–4494): one `MTL4CommitOptions` that every replay's commits carry (:4613,
    :4615). `MetalReplayEncode` (:4358) and `MetalReplaySelfTimed` (:4407) commit with the one instance their caller
    passes, `MetalReplayEncode` once a later pass and once more at its end (:4375).
- **Measured:** on 2026-11-08 every rank of the five stopped at its prefill: "prefill: Operation timed out: no feedback
  of its commits" (output_data/e2b-five-2026-11-08). Rank 0 alone, with API validation on (`MTL_DEBUG_LAYER=1`,
  design/commit-feedback-sources-2026-11-08.md §2), stopped the same way, and validation reported no error. The step
  event had reached each chunk's value, which MTL4CommandQueue.h L261 signals "after all GPU work prior to this point is
  complete", and fewer handlers than chunks had run.
- **Source:** design/commit-feedback-sources-2026-11-08.md (CF) §1: `addFeedbackHandler:` "Registers a commit feedback
  handler that Metal calls with feedback data when available." The headers do not say how many commits one handler
  serves. CF §3 measures it: "With one options instance reused, eight commits called its handler once. With one options
  instance a commit, each of the eight called its handler."
- **Write:** every commit carries an `MTL4CommitOptions` whose feedback handler was added for that commit alone, the
  form metal_recording.m `MetalCommitted` (:180–189) has. Each feedback reaches its own commit's record (struct
  Feedback's semaphore, error and GPU times; struct MetalRun's `failure`), so `feedback_wait` receives one feedback a
  commit.
