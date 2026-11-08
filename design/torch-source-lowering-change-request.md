# Change request: the torch scope's lowering from source

Code is cited at mesh 15d374b9 and metal-microbench bed56a95 where a line names no other commit.

The quoted sources were read, and each is recorded where it was read:
- design/final-lowering-sources-2026-11-06.md (FL);
- design/process-network-sources-2026-11-07.md (PN);
- design/warmup-events-handles-sources-2026-11-07.md (WE);
- design/completion-and-period-sources-2026-11-07.md (CP);
- design/modes-and-strategies-sources-2026-11-07.md (MS);
- design/worklists-and-arrival-sources-2026-11-07.md (WA);
- design/metal4-orderings-sources-2026-11-07.md (MO);
- design/msl-sources-2026-11-07.md (MSL).

Turns are named by their local time.

## The turns

> can't do any less than all 3; 
> re point 2: let's treat this as something analogous to the use of cudagraphs in pytorch where we have a foreign function api something like a python host can use, compile, and replay for programs invariant up to certain limits without recompilation or python-sided or torch-sided overhead thereafter

(2026-09-11 19:29)

> why are we letting torch control any resources (after tracing, compilation, lowering, etc, which should be AOT and produce serialized objects we cna resume without needing to repeatedly re-compile anything btw) when *we* control *our* own code and calling context?

(2026-10-28 11:45, in part)

> "[...] mesh_compile.py lowers each operator to torch's MPS kernels." did anyone ever say to do this?

(2026-11-06 13:33, in part)

> we can't even know if the code is *wrong* or *needs* to be changed if you're only using ungrounded citation-free descriptions of what he program deos without reference to any algorithms or any data structures in any paper, book, reference program, etc. saying 'torch does this' is not a usitable reference here for various reasons; we were just descirbing that torch does not have a usable mps backedn and that we are required to use altenrate irs and lowerings if contradictory or 'idiomatic' code shows up in the 'real torch mps backend'.

(2026-11-06 16:15)

> actually the repository always in the end was goign to require three scopes of support: liteRT *because i said so*, arbitrary torch lowered to mps which isn't fucking shit in performance (or arbitrarily pathologized due to bad final lowering after fxgraph or similar).

(2026-11-06 18:31, in part)

> " Today torch's eager MPS kernels are only captured, never lowered." why would we capture a torch mps kernel when torch sucks balls? what was the fxgraph thing again? i'm sort of confused why i keep seeing 'and we also have torch eager mps kernels, which are obviously bad...'. what data flow produces them? please explain in less idiomatic terms?

(2026-11-07 19:16)

> "Each function's FX graph is lowered the way PyTorch 2 §4.3 states it, one operator at a time:" find a better compiler to cite thanks

(2026-11-07 20:46)

> this is so plainly a self-inflicted harmful choice that we need to walk right back out of it. 
> 1: we hve a data structure of a partially or totally parsed call graph
> 2: all we really have to do is make it into something which is runnable by mps
> 3: mps is not a pointwise eager launch abstract computer. it is actually like, a specific api which has many well written programs it supports.
> 4: we are not here to repeat any and every bad decision torch makes in a default built in compiler. we can read about any compiler we want to and any compiler algorihtms we want to. you iwll notice this already happened for a lot of other topics? like the co-flow independent flow thing? what existingcitations have you simply not remembered orc ontextualized as related to this problem?
> 5:what further citations do they entail?

(2026-11-07 20:48)

> "- None of these produces one launch per operator. The per-operator lowering I wrote was the degenerate case: every tile one node, every node its own launch."
> 1:what is a cooperative tensor to metal?
> 2:what is a command buffer to metal?
> 3:what relationships do they have to the type annoations we already must do for collective scheduling and tp placement?
> 4: what are the equivalents to a persistent kernel in cuda et rocm in the metal apis?
> 5: under which circumstances is it an already-ordained strict fact that multiple 'stages' of consumption of a streamed tiling/tensor operation may concurrently operate with other function executions who do not mutually depend upon one another?
> 6: what api compositions does metal 'say' to use for this?
> 7: suppose we had a 'metal disassembler-reassembler' (literally, a program which turns AIR outputs of 'metal' to observable and reconfigurable latents and back to syntactically correct and semantically equivalent AIR outputs of our 'reassembler'). contrast 5 with respect to 7 with 5 with respect to 6.

(2026-11-07 21:10)

> " native GPU machine code " lets say i don't care about the idea of a fictional 'machine code' which is 'lower level than the lowest level instruction the hardware uses'. rewrite everything wrt 5,6,7 with the delusional fantasy of 'lower level than the lowest level api' coding schemes for hardware that we did not make and will never interpret in any terms other than 'what we can submit to it'.

(2026-11-07 21:17)

> "  - A function we hold only as a Metal library, such as torch's kernels or another framework's, " you should note we are now totally prohibiting the use of any 'torch metal kernels' rather than a lowering through source, any source, it can be reference soruce, it can be our source, i don't care, but we're never going to be shackling ourselves to the eager launch everything degenerate case chudsters anymore. got that? however we need to cite or ground this in non-idiomatic-terms, this will be going into at least one change request.
> "- Everywhere else, only orderings are stated. B"
> you will note that otherwise, there is nothing special about the counterfactual reassembler needed for our agenda or our performance requirements. the correct lowering of arbitrary tensor processing function libraries to *our* mps backend is simple:
> 1: we must parse callgraphs into aot forms where there are no data dependent whoopsie-doodles which can somehow change the ordering or data dependency semantics of function inputs and outputs.
> 2: we are then completely done. we know what the ordering relationships are so we can use any canonical lowering strategy for turning dependency relations into ordered vs non-ordered relations between function inputs and outputs to lower2metal.
> 3: seirously, we're done at step 2. did you think we needed a step 3 despite 'we are then completely done'?
>
> alongside what we already are fully aware we know wrt "4. Metal's equivalents of a persistent kernel in CUDA and ROCm" we should be able to pin down a specification of what it is that our compilation from 'arbitrary source language' -> 'combined tensor parallelism and collective and mps functional lowering' *ought* to do, *why* in hardware backend and algorihtmic terms there are no free choices here (the user is not getting creative adn deriving a new thing to do with abstractions. on some level all of these constraints are what the mps developer materials said to do this entire time, but more verbosely and concretely), and *what* our change request documents must strictly citea nd strictly say to entail this approach.

(2026-11-07 21:26)

> "Does the ruling also cover LiteRT's delegate kernels? " litert is like 5x faster than pytorch so it has special exceptions and special permissions to be graph captured or graph replayed or similar. if torch didn't suck we wouldn't be writing any of the code about functional lowering, would we? this would have been done days ago and we would have simply said 'we have injected collective communications and fused collective-function operations alongside tensor parallelism sharding along device for pytorch and that's all we needed since pytorch reaches roofline on mps'. but that's not true; torchmps is not an acceptable backend for research, training, inference, deployment, anything. which is why we are stuck here making torchmesh.
> otherwise whatever compose the change requset in the proposed form, using the earlier strict accounting verbatim rather than paraphrasing, compressing, or embellsihing on it on hunches

(2026-11-07 21:32)

> "A recorded kernel can't be fused, because it is fixed." take note that we could record kernels in a less pathological way if we cared through something analogous to the 'AIR -> LATENT -> AIR' reassembling transpiler implied in the user turns. but we do not necessarily have any research interest in developting this, only noting that there is *no non-idiomatic reason* for why we could not retrospectively improve arbitrary metal we found running on our machines with respect to the typing, algorithms, and data structures we are using. (are you noticing how even ismple sounding embellishments or expansions to what's present in the literature cna be crushingly wrong? such is the problem of idiomaticity.) otherwise spec section 2 sounds kinda okay instead of slippery and eluding our needs. 
> "Shall I rewrite findings 2 and 4 in the sources' own words? " sure

(2026-11-07 21:50)

> "so functions placed on the CPU now fail to compile, though until now they ran on torch's CPU kernels. " this integratino gap is a requirement to Actually Write The Compilation Stages Which Lower Arbitrary Linal To Apple Silicon Backends.
> i don't think that the user said to use an idiomatic expression liek 'all functions are lowered from source'; the non-idiomatic approach would cite the chain of IRs involved in compilation and say 'there is an IR drawn from the source [citation] which we further lower satisfying properties [citation] [citation] [citation]'.

(2026-11-07 23:54)

> i don't think *anything* is open here at all, is there? there are only non-forced errors by claudes who decided to embellish, garnish, elaborate, extend, extrapolate, or otherwise throw in extra little zingers they didn't need to write, and otherwise deviate from the standards of how change requests are formulated

(2026-11-07 23:55)

> okay so why did you give a subagent some asisgnemnt of something to do besides this exact content? what were you doing insteade? did you think that chagne request documents don't have a template or format for how they're written? address the root cause of the most recent waste of time and effort and launch implementation agents again

(2026-11-08 00:09)

## The specification

design/change-requests.md: "A change request describes what ought to be written. It uses only non-idiomatic descriptions
of work: each requirement cites the reference authors, theory, papers, books or source code that say exactly what to
write, and the code is written as that source states it. Code already in these repositories and documents Claude wrote
are not sources, since a change request exists to replace idiomatic or incomplete code with the cited method. Its body
is the turns that led to it, quoted verbatim, and where no source says what to write it says so and invents nothing."

## Requirements

Each requirement names the code it replaces or keeps, the quoted source, and what is written.

### 1. No torch kernel; every function from source

- **Replaces,** in rdma/mesh_compile.py: `_apply` (:1521); `_captured`'s recording of torch's kernels (:1775); `_kernels`
  (:1394).
- **Keeps:** the LiteRT scope's capture and replay, metal-microbench tools/e2b_replay.m `capture:` (:713) and `replay:`
  (:869).
- **Source:**
  - the operator (2026-11-07 21:26): "we are now totally prohibiting the use of any 'torch metal kernels' rather than a
    lowering through source, any source, it can be reference soruce, it can be our source, i don't care"; and
    (21:32): "litert is like 5x faster than pytorch so it has special exceptions and special permissions to be graph
    captured or graph replayed or similar."
  - J. Ansel et al., PyTorch 2, ASPLOS 2024, §1 (FL): "The framework only has visibility of a single operator at a time,
    and thus cannot automatically perform optimizations, like fusion or scheduling, that cross operator boundaries."
  - T. Chen et al., TVM, OSDI 2018, §3 (FL): "Operator fusion combines multiple operators into a single kernel without
    saving the intermediate results in memory."
  - Apple, Metal Shading Language Specification 4.1, §2.22.3 (MSL §1): "You typically don’t construct cooperative_tensor
    directly as the Layout is device specific. Instead, you use libraries such as Metal Performance Primitives (see
    section 7), a library of optimized primitives that include operators that work on tensors such as matrix
    multiplication and convolution."
  - Apple, MPPTensorOpsMatMul2d.h:207–225, macOS 26.4 SDK (FL §14b): "User can apply post processing in-register where
    GEMM output is computed using cooperative_tensor."
  - the operator, quoted in design/plan-to-bridge-change-request.md R4: "function A requires set of pages operand_a to be
    contiguous and complete before function A can begin to execute. function B requires set of pages operand_b to be
    complete but not contiguous before function B is complete. function B can work on any pages received in any order
    and can also release incomplete pages to consumer functions."
  - the operator (2026-11-08, quoted above): "this integratino gap is a requirement to Actually Write The Compilation
    Stages Which Lower Arbitrary Linal To Apple Silicon Backends."
- **Write:** no torch Metal kernel is ever submitted or recorded again. Each function, on whichever Apple silicon
  backend it is placed, the CPU included, is typed and scheduled as design/plan-to-bridge-change-request.md R3 and R4
  write it:
  - the synchronous, complete, contiguous consumer is one task with an edge from every page of its operand, and only its
    operand, typed `contiguous_input` or `contiguous_output`, is one run of pages;
  - the asynchronous, incomplete, noncontiguous consumer is one task per page, each with an edge from its own page, and
    its outputs are pages that leave as each is produced;
  - the IR carries each consumer's kind (design/aot-compilation-change-request.md A8).

### 2. The capture is a static dataflow graph

- **Keeps:** rdma/mesh_compile.py `program` (:2122), whose `_Export` (:2072) exports each function with
  `torch.export.export` (:2105).
- **Source:**
  - PyTorch, docs/source/user_guide/torch_compiler/torch.compiler_ir.md (PN §4): "Core aten IR is fully functional, and
    there is no `inplace` or `_out` variants in this opset."
  - PyTorch, docs/source/notes/cuda.md L1683 (WE §1): "Dynamic control flow (based on CPU or GPU data) is prohibited,
    unless it is based on GPU data and implemented via higher order operator"
  - E. A. Lee, D. G. Messerschmitt, "Static Scheduling of Synchronous Data Flow Programs for Digital Signal Processing",
    IEEE Trans. Computers C-36(1), 1987, §IV, p. 26 (CP §8): "The desired schedule is *admissible*, meaning that blocks
    will be scheduled to run only when data are available, and that a finite amount of memory is required."
- **Write:** a program whose dynamic control flow is based on CPU data, or on GPU data without a higher order operator,
  is not compiled.

### 3. Self-timed lowering of dependences into Metal orderings

- **Replaces:** the order of the torch scope's GPU commands, which is the order `_captured` (:1775) recorded them in.
- **Keeps,** in rdma/mesh-graph.c: `mesh_graph_completion_times` (:950) and `mesh_graph_heap_order` (:1068), called from
  `plan_schedule` (rdma/mesh-collective.c:830, :834); `mesh_graph_convert_sc` (:1263) and `mesh_graph_determine_delays`
  (:1304), called from `plan_graph` (rdma/mesh-collective.c:999–1000); rdma/mesh_compile.py `partition` (:610).
- **Source:**
  - the operator (2026-11-07 21:26): "we know what the ordering relationships are so we can use any canonical lowering
    strategy for turning dependency relations into ordered vs non-ordered relations between function inputs and outputs
    to lower2metal."
  - S. S. Bhattacharyya, S. Sriram, E. A. Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", IEEE Trans.
    Signal Processing 45(6), 1997, preprint p. 3 (MS §3): "Each processor executes the actors assigned to it in the
    order specified at compile time. Before firing an actor, a processor waits for the data needed by that actor to
    become available."
  - S. Sriram, PhD thesis, UC Berkeley, 1995, §5.4, p. 122 (CP §5a): "This is because the synchronization edges alone
    determine the interaction between processors; a communication edge without synchronization does not constrain the
    execution of the corresponding processors in any way." §5.8, p. 141: "Theorem 5.4: If G_s preserves G_ipc and the
    synchronization edges in G_s are implemented, then for each feedback communication edge e in G_ipc, the self-timed
    buffer bound of e (B_fb(e)) — an upper bound on the number of data tokens that can be present on e — is given by:"
  - Apple, MTL4CommandEncoder.h, macOS 26.4 SDK (MO §1): "Encode a barrier that guarantees that any subsequent work you
    encode in the *current command encoder*, corresponding to `beforeEncoderStages`, doesn't begin until all prior
    commands in this command encoder, corresponding to `afterEncoderStages`, completes." "This method encodes a barrier
    that guarantees that any work you encode using *subsequent command encoders*, corresponding to
    `beforeQueueStages`, don't begin until all commands you previously encode in the current encoder (and prior
    encoders), corresponding to `afterStages`, complete." "This method encodes a command that updates a ``MTLFence``
    instance after all previously-encoded commands in the current command encoder, corresponding to
    `afterEncoderStages`, complete." "Encode a command that guarantees that any subsequent work you encode via this
    current command encoder, corresponding to `beforeEncoderStages`, doesn't begin until all prior updates to the fence
    is complete." "To successfully wait for a fence update, schedule update and wait operations on the same command
    queue." Its headings: "Encodes a consumer barrier on work you commit to the same
    command queue." "Encodes a producer barrier on work committed to the same command queue." "Encodes an intra-pass
    barrier."
  - Apple, MTL4CommandQueue.h (MO §2): "Schedules an operation to signal a GPU event with a specific value after all GPU
    work prior to this point is complete." "Schedules an operation to wait for a GPU event of a specific value before
    continuing to execute any future GPU work."
  - Apple, MTLEvent.h (MO §2): "The device this event can be used with. Will be nil when the event is shared across
    devices (i.e. MTLSharedEvent)." "Synchronously wait for the signaledValue to be greater than or equal to 'value',
    with a timeout specified in milliseconds."
  - Apple, Metal Shading Language Specification 4.1, §6.10.1 (MSL §4): "A barrier function (threadgroup_barrier or
    simdgroup_barrier) acts as an execution and memory barrier. All threads in a threadgroup (or SIMD-group) executing
    the kernel must encounter the threadgroup_barrier (or simdgroup_barrier) function." §4.4 (MSL §2): "A GPU compute
    unit can execute multiple threads concurrently in a threadgroup, and a GPU can execute a separate threadgroup for
    each of its compute units."
  - Apple, MTLIndirectCommandBuffer.h (MO §3): "Dispatch threadgroups with concurrent execution"
- **Write:**
  - Assignment is the placement operand together with the sharding.
  - Order is the completion-time linear program (joint-compilation row 8), each task's length from the weighting
    operand (design/aot-compilation-change-request.md A3). It is the order `plan_schedule` (rdma/mesh-collective.c:830,
    :834) already computes, `mesh_graph_completion_times` (rdma/mesh-graph.c:950) then `mesh_graph_heap_order`
    (:1068), published with the AOT compiled static link schedule. The replay's command list is that order. The second
    linear program over the GPU's commands (metal-microbench 3b719f10, `MetalOrder`) is deleted.
  - Each dependence becomes the one Metal ordering its two ends' locations admit:
    - same threadgroup: a threadgroup barrier;
    - same encoder: an intra-pass barrier;
    - same queue: a fence;
    - other queue, device or host: an event.
  - Independent pairs get no ordering, and Metal then permits them to overlap.
  - Only the edges of a synchronization graph that preserves the inter-processor communication graph are written: "the
    synchronization edges alone determine the interaction between processors".

### 4. Consumption of received data inside functions

- **Replaces,** in the torch scope: metal-microbench metal_recording.m `MetalMaskReaders` (:1618), `MetalMaskedPass`
  (:1447) and `MetalCompactionSource` (:1596), and the later passes, `MetalLater` (:83), made at :1686–1695.
- **Source:**
  - MPI 4.1, §4.2.2, p. 120 (WA §5.3): "The MPI_PARRIVED function can be used to determine if the message data for the
    indicated partition has been received into the receive buffer. Upon success, the receiver becomes free to access
    the indicated partition (as well as any others that previously completed for that operation)."
  - Apple, Metal Shading Language Specification 4.1, §5.2.3 (MSL §3): "SIMD-groups execute concurrently within a given
    threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup
    barrier operations." §2.22.3 (MSL §1): "The cooperative_tensor represents a tensor with elements that are
    partitioned across a set of participating threads in thread memory."
  - Apple, MPPTensorOpsMatMul2d.h:296–321 (FL §14c): "It is undefined behavior if the number of SIMD groups
    dispatched does not match the number of SIMD groups that the operation was configured with." "The operation will be
    executed cooperatively by N SIMD groups. Must be used when all threads in a threadgroup are cooperatively performing
    the operation." :207–225 (FL §14b):
    "cooperative_tensor owns thread private data and divides the data for entire tensor among threads (participating in
    the scope of operation) in implementation defined manner."
  - L.-W. Chang et al., FLUX (arXiv:2406.06858), abstract (FL §10): "Flux overdecomposes communication and computation
    operations into much finer-grained operations and further fuses them into a larger kernel to effectively hide
    communication without compromising kernel efficiency." §1: "In the fused kernel, each dependent computation and
    communication tile is mapped into each thread block" §3.2: "AllGather signal checking is fused into the prologue of
    the GEMM kernel." "On the kernel side, GEMM tile computation is blocked by the function WaitSignal until the value
    contained in the signal is set to true."
  - S. Zheng et al., TileLink, §3.2.1 (FL §11): "the wait primitives carry acquire semantics, ensuring that any memory
    access following consumer(peer)_tile_wait and rank_wait cannot be executed before these wait primitives."
  - S. Zheng et al., TileLink (arXiv:2503.20313), abstract (FL §11): "TileLink decouples the design space of
    communication and computation, linking these two parts via tile-centric primitives."
  - Apple, MPPTensorOpsMatMul2d.h:158–160 (MO §4): "User can also take ownership of looping over reduction or
    k-dimension by choosing appropriate chunk size in k (called k-tile or tilek)"
- **Withdrawn** (2026-11-08). design/plan-to-bridge-change-request.md R4 writes this consumer, "one task per page, each
  with an edge from its own page", and item 1 lowers it so. Nothing waits inside a function.
