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
- **Source, 2026-11-09:** the operator: "we shall not ocnsider arbitrary python to be part of arbitrary torch;
  programmers who want to run arbitrary data dependent control flow between sections of linear algebra are to learn
  to write that sort of code on their own time and outside of our mandate and the library features we provide".
- **Write:** a program with dynamic control flow is not compiled, whether the control flow is based on CPU data or on
  GPU data, and whether or not it is written with a higher order operator (`torch.cond`, `while_loop`, `map`,
  `scan`). The exception in the PyTorch sentence quoted above is not taken. A dimension known only at run time is
  refused the same way (design/aot-compilation-change-request.md A1: "shapes are compiled").

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
### 5. An operator whose method iterates until it converges is one call of a supplier's function

The operator, in order:

> okay so we do'nt need them if they use a data-dependent control flow, and anyone who wants something like them can write their own fixed depth solver as long as we emit boring ah ah errors which tell the user in boring ah ah terms that data dependent iteration depth isn't compilable with meshlib. do you have any counterexamples of graph compilation libraries which give users a sneaky way to run divergent iterators forever with a better approach than this?

(2026-11-09 19:50, in part: the turn opens by quoting Claude's report that the lowerings "for matrix_exp, eigh, eig,
eigvals, svd, pinv, and lstsq with the gelss/gelsd drivers iterated with a WHILE on a convergence test", which is
what "them" names)

> then choose any counterexample from the literature as the lowering approach

(2026-11-09 19:51)

- **Replaces:** the lowerings mesh 199eb9c3 deleted from rdma/mesh_compile.py (`_matrix_exp`, `_jacobi_sweeps`,
  `_linalg_schur`, `_eigen_hermitian`, `_eigen`, `_singular`, `_pseudoinverse`, the gelss and gelsd branch of
  `_least_squares`), each of which repeated a body while a predicate computed from tensor data held; and that
  commit's narrowing of `_lstsq_metadata` to "driver must be gels or gelsy".
- **Source:**
  - PyTorch 2.15, `torch.linalg.lstsq`, https://docs.pytorch.org/docs/2.15/generated/torch.linalg.lstsq.html: `driver`:
    "name of the LAPACK/MAGMA method to be used." "For CPU inputs the valid values are 'gels', 'gelsy', 'gelsd,
    'gelss'." "'gelsd' (tridiagonal reduction and SVD)" "'gelss' (full SVD)."
  - PyTorch 2.15, `torch.linalg.pinv`, https://docs.pytorch.org/docs/2.15/generated/torch.linalg.pinv.html: "This
    function uses torch.linalg.svd() if hermitian= False and torch.linalg.eigh() if hermitian= True."
  - PyTorch 2.15, `torch.linalg.svd`, https://docs.pytorch.org/docs/2.15/generated/torch.linalg.svd.html: `driver`:
    "name of the cuSOLVER method to be used." "Available options are: None, gesvd, gesvdj, and gesvda."
  - LAPACK, `dgesdd`: it computes the singular value decomposition, and its `INFO` argument, when positive, means
    "DBDSDC did not converge, updating process failed."
  - Apple, Accelerate, https://developer.apple.com/accelerate: "LAPACK includes support for eigenvalue and
    singular-value problems, matrix factorization, as well as solving systems of linear equations and linear least
    squares." The macOS SDK's `vecLib.framework/Headers/clapack.h` declares `dgesdd_` (:2725), `dsyevd_` (:4817),
    `zheevd_` (:8602), `dgeev_` (:2549), `dgelsd_` (:2629) and `dgelss_` (:2638); metal-microbench `metal_native.mm`
    (:6-8) already includes Accelerate with its LAPACK interface.
- **Write:**
  - `torch.linalg.svd`, `eigh`, `eig` and `eigvals`, and `lstsq` with the gelsd and gelss drivers, are each one node
    of the graph, with fixed shapes, that calls one LAPACK driver for that problem through Accelerate. The iteration
    is inside that one function. Nothing in the graph branches or repeats on tensor data.
  - For `lstsq` the quoted page names the routine: `gelsd` and `gelss`. For `svd`, `eigh`, `eig` and `eigvals` no
    quoted source names the LAPACK routine PyTorch or JAX calls on a CPU; the header lines above show only which
    drivers Accelerate declares.
  - The routine's `INFO` is a value the node returns; a positive `INFO` is reported through the check the compiler
    already has for a numerical operation's status.
  - `torch.linalg.pinv` is the composition PyTorch states, of the node for `svd` or for `eigh`.
  - `_lstsq_metadata` takes the four drivers PyTorch names, with the output shapes PyTorch gives each.
  - `torch.linalg.matrix_exp` has no such function in LAPACK. It, and any operator whose method needs a number of
    iterations that depends on tensor data and for which no supplier's function is cited on the processor the
    placement names, is refused at export with one plain message that names the operator and says that an iteration
    depth that depends on data is not compiled. The functions cited above are the CPU's. Placement is an operand
    (the operator, quoted in design/aot-compilation-change-request.md A2: "we simply need to consume operands which
    choose these placements, letting us define and implement an optimizing chooser function *later*"): the node is
    not moved to a processor the placement did not name. Item 6 states what these operators are on the other
    processors.
  - No lowering written by mesh repeats a body a number of times that tensor data decides.

### 6. The same operators as matrix multiplications and QR factorizations, placed on every processor

The operator:

> "- The cited functions are the CPU's. Accelerate's LAPACK is the only supplier function cited for these operators. Nothing is cited for the GPU or the Neural Engine." then write a better backend which allows for the full saturation of the full resources available on apple silicon
> i don't care what other topics this sounds like it countermands; i am not patient with arbitrary waste of silicon lifespan

(2026-11-09 19:55)

> "- what matrix_exp returns when more than the fixed number of squarings would be needed;"
> a runtime error and some NaNs; these are not reasons to stop integrations of other code with other requirements which do not need this endless digression into whether we can reduce the number of error-returning compiled runtime function calls further

(2026-11-09 20:01)

- **Replaces:** item 5's refusal of `svd`, `eigh`, `pinv` and `lstsq`'s gelsd and gelss on a processor other than the
  CPU, and its refusal of `matrix_exp`.
- **Source:**
  - Y. Nakatsukasa and N. J. Higham, "Stable and efficient spectral divide and conquer algorithms for the symmetric
    eigenvalue decomposition and the SVD", SIAM J. Sci. Comput. 35(3), 2013, pp. A1325–A1349
    (https://people.maths.ox.ac.uk/nakatsukasa/publishedpdf/pub13.pdf), §1: the algorithms "are based on the polar
    decomposition", "require just the building blocks of matrix multiplication and QR factorization (without
    pivoting)", and "asymptotically minimize communication". "For an n×n matrix A, the dominant cost of QDWH-eig is
    in performing six or fewer QR factorizations of 2n × n matrices." §3.1: "We describe how to compute an invariant
    subspace of A corresponding to the positive (or negative) eigenvalues using the polar decomposition." §4:
    "Algorithm 2. QDWH-SVD: compute the SVD A = UΣV∗ of A ∈ R^{m×n} with m ≥ n. 1 Compute the polar decomposition
    A = UpH via QDWH. 2 Compute the symmetric eigendecomposition H = VΣV∗ via QDWH-eig. 3 Form U = UpV."
  - JAX, `jax/_src/tpu/linalg/qdwh.py` (https://github.com/jax-ml/jax/blob/main/jax/_src/tpu/linalg/qdwh.py): "QDWH is
    short for QR-based dynamically weighted Halley iteration." "The Halley iteration implemented through QR
    decmopositions does not require matrix inversion." It cites Nakatsukasa, Bai and Gygi, "Optimizing Halley's
    iteration for computing the matrix polar decomposition", SIAM J. Matrix Anal. Appl. 31(5), 2010. `max_iterations
    = 10` where none is given. A step is `_use_qr` ("QDWH iteration using QR decomposition.") or `_use_cholesky`
    ("QDWH iteration using Cholesky decomposition."), chosen by `CHOLESKY_CUTOFF = 100`.
  - JAX, `jax.lax.linalg.qdwh`, https://docs.jax.dev/en/latest/_autosummary/jax.lax.linalg.qdwh.html: "QR-based
    dynamically weighted Halley iteration for polar decomposition." `max_iterations`: "Iterations will terminate
    after this many steps even if the above is unsatisfied." Returns: "A four-tuple of (u, h, num_iters,
    is_converged) containing the polar decomposition of x = u * h, the number of iterations to compute u, and
    is_converged, whose value is True when the convergence is achieved within the maximum number of iterations."
  - JAX, `jax/_src/tpu/linalg/eigh.py` (https://github.com/jax-ml/jax/blob/main/jax/_src/tpu/linalg/eigh.py):
    "Symmetric (Hermitian) eigendecomposition using QDWH" "This implementation is primarily used on TPU, but it can
    in principle work on CPU and GPU also." `split_spectrum`: "The Hermitian matrix `H` is split into two matrices
    `H_minus` `H_plus`, respectively sharing its eigenspaces beneath and above its `split_point`th eigenvalue."
    "H: The `n x n` Hermitian input, padded to `N x N`." "n: the true (dynamic) size of the matrix."
    "termination_size: Recursion ends once the blocks reach this linear size." "for those platforms this algorithm
    will only do the right thing if termination_size == 1."
  - JAX, `jax.scipy.linalg.expm`, https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.expm.html:
    `expm(A, *, upper_triangular=False, max_squarings=16)`. "This uses the scaling-and-squaring approximation method,
    with computational complexity controlled by the optional `max_squarings` argument. Theoretically, the number of
    required squarings is `max(0, ceil(log2(norm(A))) - c)` where `norm(A)` is the L1 norm and `c=2.42` for
    float64/complex128, or `c=1.97` for float32/complex64."
  - JAX, `jax/_src/scipy/linalg.py` (https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py), read
    2026-11-09. `_calc_P_Q`, for float64 and complex128: `maxnorm = 5.371920351148152`,
    `n_squarings = jnp.maximum(0, jnp.floor(jnp.log2(A_L1 / maxnorm)))`, `A = A / 2 ** n_squarings.astype(A.dtype)`,
    `idx = jnp.digitize(A_L1, conds)`, `U, V = lax.switch(idx, [_pade3, _pade5, _pade7, _pade9, _pade13], A)`; for
    float32 and complex64, `maxnorm = 3.925724783138660` and `[_pade3, _pade5, _pade7]`. `_squaring`:
    `return lax.cond(i < n_squarings, _squaring_precise, _identity, c), None` under
    `lax.scan(_scan_f, R, jnp.arange(max_squarings, dtype=n_squarings.dtype))`. `expm`:
    `R = lax.cond(n_squarings > max_squarings, _nan, _compute, (A, P, Q))`, where `_nan` is
    `return jnp.full_like(A, np.nan)`.
  - JAX, `jax.lax.cond`, https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html: "However, when transformed
    with vmap() to operate over a batch of predicates, cond is converted to select()."
  - JAX, `jax/_src/tpu/linalg/eigh.py`, `_eigh_work`, read 2026-11-09: "We turn what was originally a recursive
    algorithm into an iterative algorithm with an explicit stack." "Each step of the algorithm subdivides a block
    into two subblocks whose sizes sum to the original block size." The split is
    `split_point = reductions.nanmedian(_mask(jnp.diag(ufuncs.real(H)), (b,), np.nan))`, the sizes of the two
    subblocks are `rank` and `b - rank`, where `rank` is a value `split_spectrum` returns, and the steps run under
    `lax.while_loop(loop_cond, loop_body, (agenda, blocks, eigenvectors))` with `return ~agenda.empty()`.
  - PyTorch 2.15, `torch.linalg.pinv` and `torch.linalg.lstsq`, as quoted in item 5: `pinv` "uses
    torch.linalg.svd() if hermitian= False and torch.linalg.eigh() if hermitian= True"; "'gelss' (full SVD)".
  - A. G. M. Lewis et al., "Large Scale Distributed Linear Algebra With Tensor Processing Units", arXiv:2112.09017
    (https://ar5iv.labs.arxiv.org/html/2112.09017), §I: "they essentially only multiply matrices, but are very good
    at doing so." §III.4: "Since TPUs are natively optimized for matrix multiplication, it is most natural to
    consider also tasks based on matrix multiplication". Appendix B, of an eigensolver made from its own polynomial
    iteration: "We have implemented such an eigensolver, but have found it to be lacking in both stability and
    efficiency in practice". That polynomial iteration is not the method taken here.
  - JAX, `jax.numpy.linalg.eig`, https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.eig.html: "At present,
    non-symmetric eigendecomposition is only implemented on the CPU and GPU backends."
- **Write:**
  - The polar decomposition is QDWH: `max_iterations` iterations, an operand whose default is JAX's 10, each a QR
    factorization, or a Cholesky factorization and two triangular solves, with matrix multiplications. The number of
    iterations run is that operand. Whether the iteration converged within it is a value returned with the result.
  - `eigh` is QDWH-eig: a polar decomposition splits the matrix into two blocks that share its eigenspaces below and
    above a split point, and the same is done to each block, to the depth at which blocks reach the termination
    size, an operand. A block has a compiled size, padded, and its true size is a value.
  - `svd` is Algorithm 2: the polar decomposition, the eigendecomposition of its Hermitian factor, and one product.
  - `pinv` and `lstsq` with gelsd and gelss are the compositions PyTorch states, of that `svd` or `eigh`.
  - `matrix_exp` is scaling and squaring as `_calc_P_Q`, `_squaring` and `expm` state it. The count `n_squarings`
    is a value computed from the L1 norm. There are `max_squarings` steps, an operand whose default is JAX's 16;
    step `i` gives the square where `i < n_squarings` and its input otherwise. Each `lax.cond` and `lax.switch` of
    the source is a selection among computed values, as JAX states of `cond` under `vmap`, and not a branch. Where
    `n_squarings > max_squarings` the result is NaN, as `_nan` states, and the node's status reports an error,
    through the check the compiler already has (the operator, above: "a runtime error and some NaNs").
  - Each matrix multiplication, QR factorization, Cholesky factorization and triangular solve in these is an
    ordinary node of the graph. The placement operand names its processor, on the GPU, the CPU or the Neural
    Engine, and its sharding across nodes, as for any other function; they are scheduled with everything else. No
    node of these compositions is confined to one kind of processor by this request.
  - Where the placement names the CPU for the whole operator, item 5's one call of a LAPACK driver stands.
  - `eig` and `eigvals` of a matrix that is not Hermitian stay as item 5 states them.
  - No lowering here repeats a body a number of times that tensor data decides.
- **Stops.** No quoted source says what to write for these. They are not written, and they hold up no other item
  (the operator, above: "these are not reasons to stop integrations of other code with other requirements"):
  - the eigendecomposition of a block at the termination size when that size is more than 1;
  - how many splits `eigh` makes. In `_eigh_work` the sizes of the two subblocks and the number of steps are decided
    by the matrix's values, under `lax.while_loop`. No quoted source gives QDWH-eig a number of steps or block sizes
    that tensor data does not decide. `eigh`, and the `svd`, `pinv` and `lstsq` that use it, are written on a
    processor other than the CPU once one does; on the CPU item 5 stands;
  - `eig` and `eigvals` of a matrix that is not Hermitian on the GPU or the Neural Engine;
  - which of these nodes the Neural Engine's supplier offers, and at what precision.

