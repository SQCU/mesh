# Change request: the capture, the placement of pages, completion, and the torch scope's final lowering

Code is cited at mesh f55c541c and metal-microbench d0a72fc6. metal_recording.m before d0a72fc6 is cited at
metal-microbench 0771d07a. This text replaces this file's earlier one (1d4c2ebe, 6ddf657f, 053e4017). That text took its
sources from API headers and torch's code, and its amendment's changes cited no method, which design/change-requests.md
rules out.

The sources quoted here were read and are recorded with where they were read:
- design/replay-grounding-2026-11-06.md (RG);
- design/final-lowering-sources-2026-11-06.md (FL);
- design/view-sources-2026-11-06.md (VS);
- design/packing-and-coflows-2026-11-06.md (PC).

Turn numbers are those of design/operator-turns-2026-11-04-to-06.md, numbered on past U183 for later turns.

## The turns

> who thought they get to do offset indexing or pointer googoogaagaa? a function which requires contiguous reads needs
> to be literally typed as 'contiguous_input' or 'contiguous_output' or both. all other functions must always work with
> arbitrary views of arbitrarily placed pages.

(2026-11-02 15:20, answering «Is a lease's set of rows one contiguous range in GPU address space?»)

> is this solved by MASKING or is it solved by WAITING? finally, is the LIST OF MASKED PAGE INDICES a contiguous array
> iN ITS OWN SECTION OF THE MEMORY ALLOCATION, or is it a struct which MUST BE CONTIGUOSU WITH THE PAGE IT IS ABOUT FOR
> SOME STUPID REASON?

(U41, 2026-11-03 20:35, in part)

> what did the user say to do wrt torch programs? Running any arbitrary torch program means taking torchpython, doing
> something with it s.t. the torchypython  is segmented into units which can use the library's required transport
> primitives for tensor parallelism and so on [...]
> if ther's an entire cudagraphs equivalent parser *in the library* which lets us aot replace a torchpython script with
> an IR that can be transduced into a metal runtime that isn't pathological and stupid, and the user has gathered
> citations, methods, and requirements already, even more explanation of what was specified-already needs to be written
> again *in light of this contradiciton which is only contradictory to claudes who want to write idiomatic code instead
> of composing literature with literature*.

(U159, 2026-11-06 14:03, in part)

> so can we simply use the existing mechanisms composed s.t. the most recent work presents completed schedules,
> allocation plans, etc. to these apis, resolving this apparent conflict forever ('we can either use torch generally or
> be fast' turned into 'we cna use torch generally *and also* be fast *and also* use the grounded cited scheduler and
> allocation work'). if we can, this is a nearly complete change request document (another one) all by itself.

(U164, 14:41)

> okay sure. have a subagent integrate this. this is what i expected implemetners to *do* with the previous rounds of
> assigned integration work and change requests this entire time. it was right there in teh code the entire time etc

(U165, 14:47)

> everything here works if you're using page table indices instead of trying to sneakily get access to firmware
> pointers for the dma bridge, which you should not be trying to do, ever, for obvious reasons

(U166, 14:48)

> `xs that ys refuse` are without exception idiomatic programming and are not fundamental constriants or requirements
> ot be routed around. identify why/when/how the unimplemented slop is unimplemented then considerhaving an agent
> implement the canonical algorithm/data structure/method for those scoeps rather htan creating tech debt, fall backs,
> or other idiomatic code

(U167, 16:05, answering «The torch copy_ calls in mesh_compile.py are blits the recorder refuses, so routed nodes can't
be captured yet;»)

> we can't even know if the code is *wrong* or *needs* to be changed if you're only using ungrounded citation-free
> descriptions of what he program deos without reference to any algorithms or any data structures in any paper, book,
> reference program, etc. saying 'torch does this' is not a usitable reference here for various reasons; we were just
> descirbing that torch does not have a usable mps backedn and that we are required to use altenrate irs and lowerings
> if contradictory or 'idiomatic' code shows up in the 'real torch mps backend'.

(U173, 16:15)

> yeah, no. we are here to rip out and dispose of idiomatic code in the repositories, to reach feature completeness, and
> to replace incomplete and wrong implementations with complete and sota implementations rather than regressing buggy
> incomplete code even further.

(U177, 16:38, in part, answering «I added "composed with mechanisms already in the two repositories"»)

> "- Splitting the replay at each wait: an indirect command buffer holds only draws and dispatches.' this is idiomatic
> and is not explained by saying 'there is such a thing as indirect command buffers'.
> "- Testing landed pages per threadgroup: MSL's barrier rule requires it. " what a coincidence 'once x is ruled out, y
> is required'. idiomatic as well.
> "- Ruling out placement sparse buffers: a heap's memory is always Metal's." nobody said to use the heap data structure
> for everything. the user reminded claude several turns ago today taht there is something called a cooperative tensor.
> this work is *under-cited* even if it sounds grave, respectable, and self-serious. [...]

(U185, in part)

> is it clear yet that the user has already told you how to get out of this self-limiting spiral of pointless
> failedits, which is to cite xternal algorithms, data structure, and computer science literature, both to describe what
> code already does and what the code probably *should* do? [...] you areb eing told *unsubtly* that you need to
> *accept* this problem exists (of bad and unjustified previous rounds of programming) and also *accept* that this
> approach that the user has already applied dialectically like 2-3 times now *works perfectly fine* if you paly your own
> part in it, which is discussing the state of the repository, choosing reference ltierature, documenting the literature
> in pinned explicit documents, then literally passing those literal documents in the literal template alrady shown and
> documented to subagents.

(U186, in part)

> "The capture itself has no specified method; " then find more literature and citations to pull in than saying
> 'cudagraphs'. 'we're going tom ake cudagrapsh at home' is idiomatic and more importantly names no data structure, no
> algorithm, no authors, and nothing to strictly refernce or build. [...]
> when code does osmething nonsensical that doesn't come from a citation we can say that the c urrent approach is
> idiomatic and we need to implement the referenced approach.
> "6."
> what is everyone's deal with sparse and noncontiguous memory access? [...] we need to positively name what it is they
> are doing, then positively name what it is that they are supposed to be doing, which is porgramming by view from
> homogenously accessible memory in the same plce in a cache heirarchy, which is ALL SHARED MEMORY IN THE METAL BACKENDS
> WE ARE USING. [...]
> "9. Nodes that replay alongside nodes that rerun Python."
> idiomatic output which provided no citation or justification where and when it departed, as usual for idiomatic code.

(U187, in part)

> actually paging exists because i said to use it and because every time it comes to the test, paging was always the
> correct prior constraint which enables the use of tractable nearly instant algorithms instead of unconstrained np hard
> problems. so it's right because its' right and because the ltierature keeps saying it's right, over and over and over
> and over again. [...]
> for programs that need literalyl contiguous buffers this is a problem which needs a shape selecting bin packing
> optimizing solver for which inptus and which outputs are going to go where in the memory mapping schedule used for the
> entire runtime. this is an aot compilation problem which was already described in citation for the independent flow vs
> co-flow stuff. [...]
> "- One contiguous offset per value, from the AOT memory plan." why are you saying this exaclty? cite where the
> 'everything is contiguous requirement' came from? i don't see it anywhere? idiomatic issue?
> [...] actually the repository always in the end was goign to require three scopes of support: liteRT *because i said
> so*, arbitrary torch lowered to mps which isn't fucking shit in performance (or arbitrarily pathologized due to bad
> final lowering after fxgraph or similar). does this mean we write a new mps backend? maybe it does, maybe it doesn't. i
> don't care. the task is to name algorithmically waht code does by citing external sources that aren't the code itself
> or an inlined claim. what should the code do? well you can cite other dsl/hardware pairs to analogize what missing
> properties of the final loewring are involved in fully using the mps backend's literal capabilities and the
> corresponding use of our library's collective communciation and tensor parallelism slicing. i don't want yhou to
> 'propose' something; i want you to name canonical citable approaches which are analogous and form strict choiceless
> implementation strategies by their citation.

(U188, in part)

> if we must support three backends: arbitrary torch which is aot compilable, the ltiert backend, a third thing we aren't
> doing yet, then that does not mean we pick one of torch or litert. that's gonna be pretty clear.

(U189)

## How it is implemented

As design/problems-of-14b9a757-change-request.md states it:
- **Each change is the quoted source's mechanism, as the source states it.** A statement, branch, field, wait, copy or
  allocation that the quoted text does not contain is not written.
- **A commit names the source it implements.** Nothing else explains it: no decision records, no notes of reasons.
- **A source missing here is a stop.** Where this document quotes no source for something the code needs, the
  implementer stops and names the gap. It invents nothing to fill it.

## Scope

- **Three scopes of support (U189):** arbitrary AOT-compilable torch, the LiteRT backend, and a third not yet begun.
- **What this request covers:**
  - what all three share: the capture and replay, the placement of pages, completion and the sender's release (items 1–7);
  - the torch scope's final lowering (item 8);
  - the torch scope's nodes (item 9).
- **LiteRT's scope** is tools/e2b_replay.m's, and it changes here only through what it shares.

## Claude's proposal

### 1. The capture records every call, and the replay issues every call

- **Code:**
  - At 0771d07a, metal_recording.m `MetalBlitPreparation` runs a fill once, during the capture, and refuses every other
    blit command: `MetalRefuseInvocation(activeCapture, invocation, @"blit command")` (:3032). The replay holds dispatches
    only, in indirect command buffers.
  - d0a72fc6 records copies and fills "where a recorder records blits (MetalBlit); no recorder does yet". It also marks
    "a dispatch whose pipeline lacks supportIndirectCommandBuffers" as kind direct (its commit message).
- **Source:**
  - **apitrace** (github.com/apitrace/apitrace at 3a0ee82a, MIT-style license), README: "trace OpenGL, Direct3D, and
    DirectDraw APIs calls to a file;" and "replay OpenGL and Direct3D calls from a file;" (RG §B.1).
  - **GFXReconstruct** (github.com/LunarG/gfxreconstruct at f9b0fb0d, MIT), README: "The GFXReconstruct project
    provides tools for the capture and replay of graphics API calls, allowing the graphics commands executed by an
    application to be recorded to a file that may later be replayed to reconstruct the graphics-specific behavior of the
    captured application." (RG §B.1)
  - **Tracing:** R. Frostig, M. J. Johnson, C. Leary, "Compiling machine learning programs via high-level tracing",
    SysML 2018: "On a trace cache miss, JAX executes the corresponding Python function and traces its execution into a
    graph of primitive functions with static data dependencies." (RG §B.1)
  - **The display file:** T. H. Myer, I. E. Sutherland, "On the Design of Display Processors", CACM 11(6), 1968, p. 411:
    "The channel takes successive data words from a display file in core until the word count goes zero, at which point
    the central processor restarts the channel at the beginning of the display file." (RG §B.1)
  - **Inspector and executor:** J. Saltz, R. Mirchandaney, K. Crowley, "Run-time Parallelization and Scheduling of
    Loops", ICASE Report 90-34, 1990: "(1) inspector procedures that perform execution time preprocessing and (2)
    executors or transformed versions of source code loop structures. These transformed loop structures carry out the
    calculations planned in the inspector procedures." (RG §B.1)
  - **CUDA stream capture:** NVIDIA, CUDA Runtime API, `cudaStreamBeginCapture`: "When a stream is in capture mode, all
    operations pushed into the stream will not be executed, but will instead be captured into a graph, which will be
    returned via cudaStreamEndCapture." Its `cudaGraphNodeType` lists "GPU kernel node", "Memcpy node", "Memset node" and
    "External event wait node" (read 2026-11-06).
  - **How Metal 4 writes a copy and a fill:** MTL4ComputeCommandEncoder.h:323, "Encodes a command that copies data from a
    buffer instance into another."; :456, "Encodes a command that fills a buffer with a constant value for each byte."
- **Change:**
  - The capture records every call the captured program makes on the recorder's device, queues and command buffers,
    copies and fills included, as apitrace and GFXReconstruct record API calls "to a file".
  - The replay issues the recorded calls in their recorded order. A copy is issued with MTL4ComputeCommandEncoder's copy
    command and a fill with its fill command, as a captured graph holds memcpy and memset nodes beside its kernel nodes.
  - Deleted: the refusal, the fill run during the capture, d0a72fc6's blit mode, and its direct kind.

### 2. A value's copies, the replay's period, and how far ahead execution t starts

- **Code:**
  - metal_recording.m `MetalReplayPeriod` takes the period as the least common multiple of every placed operand's
    copies, every received operand's 4u and the step word's copies.
  - tools/e2b_replay.m:77 `#define HOST_LAG 2u`; rdma/mesh_compile.py `HOST_LAG = 2`. e2b's function has `lead` 1.
    rdma/mesh_compile.py `_replayed` passes `lead` 0 to `MetalRecorderRun`.
- **Source:**
  - **M. Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", PLDI 1988, §2.3, pp. 322–323:**
    - "If the lifetime of a variable is l, and an iteration is initiated every s cycles, then at least ⌈l/s⌉ number of
      values must be kept alive concurrently, in that many locations."
    - "If each variable v_i is allocated its minimum number of locations, q_i, the degree of unrolling is given by the
      lowest common multiple of {q_i}. Even for small values of q_i, the least common multiple can be quite large and can
      lead to an intolerable increase in code size. The code size can be reduced by trading off register space. We
      observe that the minimum degree of unrolling, u, to implement the same schedule is simply max_i q_i. This minimum
      degree of unrolling can be achieved by setting the number of registers allocated to variable v_i to be the
      smallest factor of u that is no smaller than q_i".
    (RG §B.2)
  - **S. S. Bhattacharyya, S. Sriram, E. A. Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", IEEE Trans.
    Signal Processing 45(6), 1997:** "Each processor executes the actors assigned to it in the order specified at
    compile time. Before firing an actor, a processor waits for the data needed by that actor to become available." Eq.
    (1): "start(v_i, k) ≥ end(v_j, k – delay((v_j, v_i)))". (RG §B.6)
  - **design/aot-compilation-change-request.md, "Appendix: the citations the implementation omitted or ignored", item
    2:** "The IPC edges are the plans' transfers, each with C3.3's u copies as its delay".
- **Change:**
  - A value has q = ⌈l/s⌉ copies, l its lifetime and s the interval between executions' starts. A host-written input is
    such a value, so `HOST_LAG` is its q.
  - The replay's period is u = max over the values of q. A value's copies are the smallest factor of u no smaller than
    its q.
  - Execution t's requests start once execution t − d has ended, d being the delay of the transfer's IPC edge, its u
    copies. `lead` is that delay, in `MetalRecorderRun`'s terms.
  - Deleted: the least common multiple, the 4u period, and the constants 2, 1 and 0.

### 3. Reading data a remote writer lands

- **Code:**
  - metal_recording.m `MetalMaskReaders`: every threadgroup of a command that reads a received operand compares every
    one of the operand's phase fields. Threadgroups that fail return and are compacted into a list, for a later pass
    after `waitForEvent:value:`.
  - The masked pass is written from the reader's MSL source, and the code raises where none was recorded.
  - The compacted list holds threadgroup indices. RG §B.3: "There is no "list of masked page indices" in the code".
  - d0a72fc6 makes a reader with no recorded source a whole later pass.
- **Source:**
  - **R. Nasre, M. Burtscher, K. Pingali, "Data-driven versus Topology-driven Irregular Computations on GPUs", IPDPS
    2013:** "In topology-driven implementations, all nodes are assumed to be active so the operator is applied everywhere
    even if there is no work to do at some nodes" (p. 1); "data-driven implementations are organized around a worklist"
    (p. 2); "The threads in a thread block can compute the total amount of work to be added by all threads in the thread
    block using a prefix sum (in shared memory)" (p. 3). (RG §B.3)
  - **MPI 4.1, Partitioned Point-to-Point Communication (PDF ch. 4, §4.2.1–4.2.2, pp. 119–120):** "MPI_PREADY is a
    send-side call that indicates that a given partition is ready to be transferred." "The MPI_PARRIVED function can be
    used to determine if the message data for the indicated partition has been received into the receive buffer. Upon
    success, the receiver becomes free to access the indicated partition". (RG §B.3)
  - **J. Dong et al., "Flex Attention: A Programming Model for Generating Optimized Attention Kernels" (arXiv:2412.05496),
    §4.2:** "Concretely, BlockMask contains two tensors, a kv_num_block of shape B × H × Num_Row and a kv_indices of shape
    B × H × Num_Row × Num_Col." (read 2026-11-06)
  - **The operator:** U41, above. 2026-09-08: "you're implementing data flow as control flow instead of using masks,
    indices, and explicit data flow data structures".
- **Change:**
  - The landed pages are a worklist, as Nasre et al. organize a data-driven computation. Each consumer row has a count
    and a dense array of the page indices that have arrived, in its own section of the region, in the form of BlockMask's
    `kv_num_block` and `kv_indices`.
  - A page enters its row's list when it has arrived, which is MPI_PARRIVED's per-partition test. A sent range is ready
    by MPI_PREADY's per-partition readiness, which keeps d0a72fc6's one received word and one supplied word per range.
  - A reader's threadgroups work on the pages their row's list holds, and the topology-driven comparison of every phase
    field by every threadgroup is deleted.
  - **Stop:** a kernel with no MSL source can't be given this test. In the torch scope those are torch's MPS kernels,
    which item 8's lowering replaces (U173). Until item 8 exists, a routed node of the torch scope stops here.
    d0a72fc6's whole later pass for such a reader is deleted.

### 4. Completion: execution t's waits are those of its started requests

- **Code:** rdma/mesh-metal.m `waits_run` and `mesh_metal_waits_start` walk repetitions 0 to R − 1, with R fixed when
  the waiter starts. A wait ends with ETIMEDOUT after MESH_PLATFORM_NS without progress.
- **Source:**
  - **MPI 4.1 §3.9:** "Often a communication with the same argument list (with the exception of the buffer contents) is
    repeatedly executed within the inner loop of a parallel computation. In such a situation, it may be possible to
    optimize the communication by binding the list of communication arguments to a persistent communication request once
    and then repeatedly using the request to start and complete operations." And: "The request becomes active once the
    call is made."
  - **D. P. Reed, R. K. Kanodia, "Synchronization with Eventcounts and Sequencers", CACM 22(2), 1979, p. 116:** "An
    eventcount can be thought of as a nondecreasing integer variable." "we define a primitive await(E, v) that suspends
    the calling process until the "value" of the eventcount E is at least v." (RG §B.4)
- **Change:** execution t's waits are the awaits of the requests started for execution t, each awaited once its start
  has been made. The waiter has no count of executions fixed before the first.

### 5. The sender's release: every access to the supplied word has the word's size

- **Code:**
  - metal_recording.m `MetalRelease` issues a seq_cst fence, then `atomic_fetch_add_explicit` on a 32-bit `atomic_uint`
    at the low half of the 64-bit supplied word. The bridge reads the word with a 64-bit load (RG §B.5).
  - In Metal's metal_atomic header (GPUCompiler 32023.864), `atomic_fetch_add_explicit` takes device int and uint (and
    float) only (:424–437). A device `ulong` has `atomic_max` and `atomic_min` (:689, :713).
- **Source:**
  - **S. Flur et al., "Mixed-Size Concurrency: ARM, POWER, C/C++11, and SC", POPL 2017, p. 2:** "the effective type rules
    of the ISO C standard prohibit mixed-size use of atomics." (RG §B.5)
  - **PTX ISA §8.8:** a release pattern is "a release or acquire-release memory fence followed by a strong write on M in
    program order" (RG §B.5).
  - **design/disjoint-pages-change-request.md C3.7 (:938):** "A supplied word has one writer, its range's producer, which
    stores t + 1 after its last write in execution t."
- **Change:**
  - The release is the fence, then a 64-bit write of t + 1: `atomic_max_explicit` on the word as a device `ulong`. For
    the word's one writer, whose values only rise, this writes t + 1, since max(t, t + 1) = t + 1.
  - Every access to the word is 64-bit.

### 6. Placement: pages, with runs only for contiguous-typed functions

- **Code:**
  - rdma/mesh-collective.c `plan_place` places every value page by page, with `mesh_graph_chains` (Dilworth's theorem,
    by Fulkerson's matching) and `mesh_graph_fat_cover` (Mukerji 1994 §3.2) (PC).
  - rdma/mesh-dataflow.c `mesh_view_compose` reserves a range and remaps every typed operand's rows over it with
    `mach_vm_remap`. That comes from design/plan-to-bridge-change-request.md R2 as 4e78f6f4 wrote it (2026-11-05):
    "No value is placed as one run. So greedy by size (Pisarchyk and Lee), and the contiguous-block problem it
    approximates [...], have no use here." Its precedent is the repository's own `ncclMeshPageTableCompose`, which is
    not literature (RG §A).
  - **The solver of 2026-11-02:**
    - It placed a contiguous-typed function's values as runs by TensorFlow Lite's `GreedyBySizeAssignment`, ported.
      Then it placed every other value page by page: c491c514, f56e59c6, f09d353c and e172a931 ("the greedy-by-size
      port lives in mesh-graph.c, a delimited section under TensorFlow's Apache-2.0 notice").
    - 0b3e78d7 ("mesh returns to 3981ca80", 2026-11-03) deleted it. Tag `pre-return-20261103` holds it.
- **Source:**
  - **The operator:** 2026-11-02, above (the typing); U188, above (paging, and the solver for contiguous buffers).
  - **Y. Pisarchyk, J. Lee, "Efficient Memory Management for Deep Neural Net Inference", arXiv:2001.03288, 2020, §5.2
    "Greedy by Size for Offset Calculation":** "We first iterate through tensor usage records in non-increasing order by
    their size (L.1,6). For each record, we check already assigned tensors whose usage intervals intersect with that of
    the current tensor (L.10–13) to find the smallest gap in memory between them such that current tensor fits into that
    gap (L.9,14–17). If such a gap is found, the current tensor is allocated to this gap. Otherwise, we allocate it after
    the rightmost tensor whose usage interval intersect with that of the current tensor (L.19–20)." (VS)
  - **T. Kilburn, D. B. G. Edwards, M. J. Lanigan, F. H. Sumner, "One-level storage system", 1962 (Bell and Newell
    reprint, 1971):** "Hence a system has been devised to make the core drum store combination appear to the programmer
    as a single level of storage, the requisite transfers of information taking place automatically." (VS)
  - **design/aot-compilation-change-request.md A8:** "each consumer's kind (R4: a join for function A, a task per page
    for function B)".
- **Change:**
  - A function's operand is one run only where the function is typed `contiguous_input` or `contiguous_output` (the
    operator, 2026-11-02). Every other function works on views of its operands' pages, named by page-table index (U166).
  - The AOT memory schedule places the runs first, by Greedy by Size for Offset Calculation, as f09d353c ported
    TensorFlow Lite's `GreedyBySizeAssignment`. It then places every other value page by page, by the chains and the fat
    cover.
  - A run's transfers are consumed whole, as R4's join; every other operand's pages are R4's tasks per page.
  - Deleted: `mesh_view_compose`'s remapping of typed operands. R2's "No value is placed as one run" no longer holds
    for contiguous-typed operands.

### 7. Lifetimes: from the first write to the last read

- **Code:** rdma/mesh_compile.py `program()` gives every value of a capture the capture's whole range of levels:
  `value.begin, value.end = 0, max(levels, 1) - 1`.
- **Source:** design/problems-of-14b9a757-change-request.md item 7: "Every block's item arrives at the first write and
  departs at the last read of the values whose bytes it holds, in the program's order." Its item 14 colors those
  intervals as cyclic intervals (Hendren, Gao, Altman and Mukerji, CC 1992; Mukerji 1994 §3.2).
- **Change:** a captured buffer's interval runs from the level of the first recorded command that writes it to the level
  of the last that reads it. Only the program's invariants span the whole program.

### 8. The torch scope's final lowering

- **Code:**
  - The torch scope runs torch's MPS kernels, one ATen operator at a time, and the capture records them as they are.
  - Of the ATen operators, only `aten::mm` runs the engine's `dense_tensor_f16` (`matmul2d` with cooperative tensors;
    mesh 03297e52).
- **Source:** all read and quoted in FL.
  - **J. Ansel et al., PyTorch 2, ASPLOS 2024, §1:** "The framework only has visibility of a single operator at a time,
    and thus cannot automatically perform optimizations, like fusion or scheduling, that cross operator boundaries."
  - **T. Chen et al., TVM, OSDI 2018, abstract:** "TVM solves optimization challenges specific to deep learning, such as
    high-level operator fusion, mapping to arbitrary hardware primitives, and memory latency hiding."
  - **J. Ragan-Kelley et al., Halide, PLDI 2013, §1:** "decoupling the algorithm definition from its execution strategy".
  - **P. Tillet, H. T. Kung, D. Cox, Triton, MAPL 2019, §5.2.1:** "Nested tiling strategies … aim at decomposing tiles
    into micro-tiles and eventually nano-tiles in order to fit a machine's compute capabilities and memory hierarchy as
    tightly as possible."
  - **B. Spector et al., ThunderKittens (arXiv:2410.20399), §1:** "any high performance framework must prioritize keeping
    tensor cores at high utilization whenever possible."
  - **C. Cecka, CuTe layout algebra (arXiv:2603.02298), abstract:** "These instructions prescribe fixed, often complex
    data layouts that must be correctly propagated through the entire execution pipeline to ensure both correctness and
    optimal performance."
  - **T. Dao et al., FlashAttention, NeurIPS 2022, abstract:** "an IO-aware exact attention algorithm that uses tiling to
    reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM."
  - **Y. Xu et al., GSPMD (arXiv:2105.04663), §1:** "GSPMD will run compiler passes that complete the sharding
    specification on the entire computation graph, and transform it into a mathematically equivalent, parallelized
    computation to run on each device."
  - **N. Shazeer et al., Mesh-TensorFlow, NeurIPS 2018, abstract:** "A Mesh-TensorFlow graph compiles into a SPMD program
    consisting of parallel operations coupled with collective communication primitives such as Allreduce."
  - **L.-W. Chang et al., FLUX (arXiv:2406.06858), abstract:** "Flux overdecomposes communication and computation
    operations into much finer-grained operations and further fuses them into a larger kernel to effectively hide
    communication without compromising kernel efficiency."
  - **S. Zheng et al., TileLink (arXiv:2503.20313), abstract:** "TileLink decouples the design space of communication and
    computation, linking these two parts via tile-centric primitives."
  - **A. Jangda et al., CoCoNet, ASPLOS 2022, §1:** "we need to schedule the MatMul kernel to compute chunks in the order
    the AllReduce kernel communicates them."
  - **Triton-distributed (arXiv:2504.19442), §2.1:** "The core of our programming model includes three concepts:
    symmetric memory, signal exchange, and async-task."
  - **Metal Performance Primitives, MPPTensorOpsMatMul2d.h:216–218 (macOS 26.4 SDK):** "cooperative_tensor owns thread
    private data and divides the data for entire tensor among threads (participating in the scope of operation) in
    implementation defined manner." The same header: "User can apply post processing in-register where GEMM output is
    computed using cooperative_tensor."
- **Change:**
  - The torch scope's kernels are those of a final lowering of each compiled function's graph that has, on Metal, the
    properties these sources state:
    - operators fused across their boundaries (PyTorch 2, TVM);
    - tiles fitted to the machine's compute capabilities and memory hierarchy (Triton, ThunderKittens, FlashAttention),
      held in cooperative tensors (MPP);
    - layouts propagated through the whole pipeline (CuTe);
    - the algorithm kept apart from its schedule (Halide);
    - the sharding completed over the whole graph into one SPMD program with the library's collectives (GSPMD,
      Mesh-TensorFlow);
    - communication cut into tiles and fused or ordered with the compute that consumes it (FLUX, TileLink, CoCoNet,
      Triton-distributed).
  - torch's MPS kernels are not this lowering's kernels (U173).
  - **Stop:** these sources state the properties, not which IR carries the lowering on Metal. That is the subject of the
    next change request, and nothing is written for it from this one.

### 9. Every node replays

- **Code:** mesh fcf5d18a: a node where a declared transfer names one of its values runs its steps, calling torch from
  Python at every execution, while the other nodes replay. It cites no source.
- **Source:**
  - **design/aot-compilation-change-request.md A1:** "So every function of a torch program is captured and compiled
    before the program runs, the same at every rank, and nothing runs eagerly between its compiled calls."
  - **NCCL User Guide, "Collective Operations":** "Collective operations have to be called for each rank (hence CUDA
    device), using the same count and the same datatype, to form a complete collective operation. Failure to do so will
    result in undefined behavior, including hangs, crashes, or data corruption."
  - **NCCL User Guide, "Using NCCL with CUDA Graphs":** "Starting with NCCL 2.9, NCCL operations can be captured by CUDA
    Graphs."
- **Change:** every node runs its captured program by replay, its transfers included. fcf5d18a's split is deleted. In
  the torch scope this waits on items 3 and 8.

## What goes from the earlier text and its implementation

- The earlier items 4 and 5 and the 053e4017 amendment are replaced by items 1–9.
  - Their placement sparse buffers, the heap argument and the single-VM-region point go. So do the splitting of the
    replay at waits, the direct encoding of dispatches, and the source-less whole pass. None of them cites a method.
- From d0a72fc6:
  - **Kept:** the one received word and one supplied word per range (item 3).
  - **Deleted:** the direct kind, the blit mode and the whole later pass for a source-less reader (items 1 and 3).
- From mesh 03297e52:
  - **Kept:** the capture once at make() and the replay at every execution (item 1).
  - **Replaced:** the registration of torch's MPS kernels as the torch scope's kernels, by item 8.

## Stops

- Item 3, for kernels with no MSL source, until item 8.
- Item 8, which IR carries the torch scope's final lowering.
- Item 9 in the torch scope, until items 3 and 8.
