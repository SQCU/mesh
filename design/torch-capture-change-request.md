# Change request: capture typed work and lower its storage and execution

### 1. Capture returns commands and their operands

The recorder captures the program's API calls, arguments and initial read-before-write buffer contents
(apitrace; GFXReconstruct; RenderDoc capture/replay). The typed provider supplies numerical bodies, bindings,
indexing maps and effects through the LiteRT/ML Drift/Tint interface specified by composition.
Scheduling transforms that retained program, and final lowering emits its compiled order and bindings.

### 2. Analysis determines invocation copies

The graph module computes the maximum cycle mean and feedforward buffer bound `Bff(e)` from the IPC graph and
execution-time estimates (Bhattacharyya, Sriram and Lee 1997, §§4.2–4.3). Its binary search starts from the bound
in equation (6) and retains the estimated iteration period when adding the reverse delay edge.
Lam's modulo variable expansion (1988) supplies one location for each simultaneously live occurrence.
Preparation binds those locations; execution `t` uses its residue after the previous use of that location completes.

### 3. Received data supplies masks and worklists

The compiler derives workgroup accesses from typed indexing maps using Lee's index-space analysis
(*Virtualizing Data Parallel Systems*, 2015, §3.2.2) and Halide's bounds analysis (2013, §4.2).
Received-page presence and the declared accesses produce a mask. Blelloch's scan and Harris–Sengupta–Owens
compaction produce the remaining worklist and indirect dispatch length. Hall et al. 1997 completion-time
priorities order its tasks through the shared graph module. The emitted operations consume these explicit values
(MPI 4.1 §4.2 partitioned arrival; Nasre, Burtscher and Pingali, IPDPS 2013).

### 4. Completion yields the invocation outcome

Persistent requests follow MPI 4.1 §3.9's create/start/complete/free lifecycle. Reed and Kanodia's eventcounts
(1979) supply monotonically increasing completion thresholds. Metal queue signals and per-commit feedback
publish the completed invocation; the client's bounded wait observes its result and status
(Apple `MTLEvent.h`, `MTL4CommandQueue.h`, `MTL4CommitFeedback`; Linux `complete_all` for failure completion).

### 5. Publication uses a 64-bit release operation

After its numerical writes, the producer issues the release fence and a single 64-bit `atomic_max_explicit`
publication of `t+1`. The bridge consumes that word's threshold (PTX ISA §8.8 release pattern; Apple
`metal_atomic` declarations for `atomic_ulong`; Flur et al., *Mixed-Size Concurrency*, POPL 2017).

### 6. Contiguous operands receive solved offsets

The allocator takes one size/lifetime record per expanded operand occurrence. Pisarchyk and Lee's Greedy by Size
for Offset Calculation (2020, §5.2), as TensorFlow Lite's `GreedyBySizeAssignment`, assigns offsets and total
arena size. The overlap predicate uses cyclic intervals on the unrolled executions (Hendren et al., ACAPS Memo 33,
§2.2; Mukerji 1994; Dagor `do_arcs_overlap`). The resulting arena and offsets bind contiguous inputs and outputs.
Other declared views use their page mappings (plan-to-bridge R2).

### 7. Lifetime analysis returns actual use intervals

The graph module derives a value's first definition and final read from scheduled operations (Lam 1988;
ExecuTorch `update_tensor_lifetime`; TensorFlow Lite `arena_planner.cc`). Input intervals begin at entry and
result intervals extend through the output boundary (Rau et al., *Register Allocation for Software Pipelined
Loops*, PLDI 1992). Transfer endpoints remain live through their actual completion (MPI 4.1 §3.7).
The allocation algorithms consume those intervals and retain their results for lowering and archive.

### 8. Source lowering fuses the numerical and communication program

The target compiler consumes typed operations, schedule, layouts and access relations. TVM operator fusion,
Halide algorithm/schedule lowering, Metal Performance Primitives' cooperative tensors, and the FLUX/TileLink/CoCoNet
communication-fusion mechanisms produce the target program (source-lowering and composition).
The compiled prologues and epilogues consume the same placed operands and dependency results as transport.

### 9. Every participating node executes its compiled program

Each rank invokes its compiled numerical and collective operations through the completed native entry,
following the graph-captured collective semantics of NCCL's CUDA Graph support and the operator's all-node
execution instruction (this file's item 9 at `cdd44406`). The archive restores that program and its bindings
through AOT A9's serialized operand.

## Execution-time estimates

The target lowering supplies operation counts, bytes read/written and execution mapping. XLA's GPU performance
model combines compute time, memory time and launch overhead; Roofline supplies machine rates and ceilings
(Williams, Waterman and Patterson 2009, §§3–4; XLA `GpuPerformanceModelBase`). The result `t(v)` is an explicit
operand of item 2 and AOT A3. Unequal-shards items 6/7/14 consume its size-dependent cost terms.
