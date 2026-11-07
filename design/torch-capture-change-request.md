# Change request: the capture, the placement of pages, completion, and the torch scope's final lowering

Code is cited at the first implementation's heads, mesh aca4f783 and metal-microbench 63b9f015, where an item says no
other commit. This text replaces this file's text of efa3ad8c and e8d1296c. As U190 has it, the first implementation's
differences are revisions of this text, and so are U191's and U192's corrections of Claude's account of them.

The sources quoted here were read, and each is recorded with where it was read:
- design/replay-grounding-2026-11-06.md (RG);
- design/final-lowering-sources-2026-11-06.md (FL);
- design/view-sources-2026-11-06.md (VS);
- design/packing-and-coflows-2026-11-06.md (PC);
- design/capture-replay-sources-2026-11-07.md (CS);
- design/completion-and-period-sources-2026-11-07.md (CP);
- design/worklists-and-arrival-sources-2026-11-07.md (WA);
- design/lifetimes-and-placement-sources-2026-11-07.md (LP).

Turn numbers are those of design/operator-turns-2026-11-04-to-06.md, numbered on past U183 for later turns.

## The turns

> who thought they get to do offset indexing or pointer googoogaagaa? a function which requires contiguous reads needs
> to be literally typed as 'contiguous_input' or 'contiguous_output' or both. all other functions must always work with
> arbitrary views of arbitrarily placed pages.

(2026-11-02 15:20, answering «Is a lease's set of rows one contiguous range in GPU address space?»)

> "- The write-after-read and write-after-write hazards disappear by construction.
> - The only hazard left is read-after-write: the program's own data edges, which the plan already holds and the
>   transport's completions already signal."
> is this solved by MASKING or is it solved by WAITING?
> finally, is the LIST OF MASKED PAGE INDICES a contiguous array iN ITS OWN SECTION OF THE MEMORY ALLOCATION, or is it a
> struct which MUST BE CONTIGUOSU WITH THE PAGE IT IS ABOUT FOR SOME STUPID REASON?

(U41, 2026-11-03 20:35, in part; design/disjoint-pages-change-request.md:250–255)

> then your code isn't correct for deliberately avoiding demonstrating the requirements of the change request.

(U48, 11-03 23:00, in part)

> function A requires set of pages operand_a to be contiguous and complete before function A can begin to execute.
> function B requires set of pages operand_b to be complete but not contiguous before function B is complete. function B
> can work on any pages received in any order and can also release incomplete pages to consumer functions. what kind of
> function is function A and what kind of function is function B? both are realistic tensor processing functions which
> are used in many widely deployed functions.

(U71, 11-04 13:19, in part)

> "e2b stays all function A, provisionally."
> this is provisionally wrong as it provisionally establishes a provisional test which is provisionally not using the
> provisional interfaces we just provisionally wrote.

(U90, 11-04 19:19, in part)

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

> sure, implement this, looks citey enough and any substantial differences should be revisions to this coverage of work

(U190, 22:38, answering this text at efa3ad8c)

> per threadgroup pages is a made up idea that has nothing to do with a citation. purely idiomatic, purely fictional
> problem. circle back and figure out what you're trying to desribe as a problem

(U191, 2026-11-07 00:19, answering Claude's draft revision of item 3)

> "Item 3 says "each consumer row has a count and a dense array of the page indices that have arrived". " idiomatic or
> citation sourced? a garnish added after something was authorized on the premise it was only a list of citations and
> plain statements to implement feature q through algorithm a_q?
> "- Code: MetalMaskReaders treats every command that reads a received operand the same way."
> what pattern of algorithm is htis in the reference literature? is there any non-idiomatic definition? does this have
> any purpose or rqeuirement that entails it?
> "  - The recorder never sees the consumer's kind. MESH_JOIN and MESH_PAGES exist only in the plan
> (mesh-collective.h:55–56, e2b_replay.m:609, mesh_compile.py:1358), not in metal_recording.{h,m}."
> this is perhaps a hint that the implementation of the coflow and independent flow approach was a guide for how to
> reuse the same IRs made for deciding such matters to produce an IR related to data processing with similar properties
> and results. it's not my fault that this was not literature checked and proposed especially with the ltieral example
> of how this approach was already used in the codebase (?)
> "With the bridge off limits,"
> was this idiomatic? an invented or fictional problem that didn't need to be presented?
> "- Recorded kernels are function A. A LiteRT or torch-MPS kernel reads its operand as one run, so by U71's words its
> reader just waits. That covers all of e2b's readers today."
> idiomatic + doesn't make sense + ignores the reference algorithms + user actually said literally the opposite several
> times in several user turns not that anyone reads them
> "- B's form applies only to kernels that take page indices. "
> 'kernels that take page indices' this sounds idiomatic especially since the assistant turns which constructed function
> a/b defined several functions which are always b-flavored and are used in the e2b litert implementation. find them,
> soruce the reasons for these respoonses, write a better change request document which adheres to the required style,
> fill in more citations and displace topics that get idiomatic responses in previous rounds wtih simple citations and
> requirements, etc.

(U192, 00:29)

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
  - what all three share: the capture and replay, the reading of landed pages, completion, the sender's release, the
    placement of pages and their lifetimes (items 1–7);
  - the torch scope's final lowering (item 8);
  - the torch scope's nodes (item 9).
- **LiteRT's scope** is tools/e2b_replay.m's and tools/prepare_e2b.py's, and it changes here only through what it
  shares.

## Claude's proposal

### 1. The capture records every call, and the replay issues every call

- **Code:**
  - Before the first implementation, metal_recording.m `MetalBlitPreparation` ran a fill once, during the capture, and
    refused every other blit command (0771d07a, :3032). The replay held dispatches only, in indirect command buffers.
  - metal-microbench 109b11a3 records copies and fills, and the replay issues them in their recorded order between its
    indirect command buffers' ranges, by MTL4ComputeCommandEncoder's copy and fill.
  - Still refused during a capture at 63b9f015 (metal_recording.m):
    - a compute-encoder call the recorder doesn't record (:3125–3131);
    - a blit-encoder call other than `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` and
      `fillBuffer:range:value:` (:3153–3157);
    - a command-buffer call whose selector begins `encode` or names a `CommandEncoder` (:3211–3214).
  - `encodeSignalEvent:value:` is passed to the command buffer during a capture and is not recorded (:3183–3184).
  - The recorder's device sets `supportIndirectCommandBuffers` for three of MTLDevice.h's six compute-pipeline creation
    methods, the synchronous ones (:2554–2569; `MetalCreatePipeline`, :491–497). A dispatch of a pipeline made any other
    way raises at replay: "Pipeline does not support indirect commands" (:3878).
  - A function is captured at its first execution: e2b's `capture:` block of each phase (tools/e2b_replay.m:1231);
    rdma/mesh_compile.py `_captured` (:3178).
  - Dawn clears a buffer at its first use, once: `Buffer::InitializeToZero` calls `ClearBuffer` and then
    `SetInitialized(true)`, and `ClearBuffer` issues `fillBuffer:range:value:` (Dawn 79794682, BufferMTL.mm:287–294,
    :304–306; CS §5). Since 109b11a3 the replay repeats those fills at every execution.
- **Source:**
  - **apitrace** (3a0ee82a, MIT), README: "trace OpenGL, Direct3D, and DirectDraw APIs calls to a file;" and "replay
    OpenGL and Direct3D calls from a file;" (RG §B.1). retrace/retrace_main.cpp:820: "loop N times (N<0 continuously)
    replaying final frame." (CS §2)
  - **GFXReconstruct** (f9b0fb0d, MIT):
    - README: "The GFXReconstruct project provides tools for the capture and replay of graphics API calls, allowing the
      graphics commands executed by an application to be recorded to a file that may later be replayed to reconstruct
      the graphics-specific behavior of the captured application." (RG §B.1)
    - USAGE_desktop_Vulkan.md:1141–1143: "Replay proceeds normally up to and including frame `N`, and then replays
      frame `N` again for each iteration." (CS §1)
    - docs/GFXReconstruct Architectural Principles.md:89: "If a call truly cannot be represented in its original form
      wholly or piecewise, that case must be documented as an exception." :102: "Capture should avoid changing the
      content of the application’s graphics commands.  Some exceptions are necessary in order to save information
      required for replay or modify data so that replay is possible." (CS §7–8)
    - framework/encode/vulkan_capture_manager.cpp:889–896, adding the capture-replay flag to a buffer's create info:
      "We modify only a copy to prevent the modified pCreateInfo from being written to the capture file." (CS §8)
  - **RenderDoc** (34fbfb43, MIT), docs/behind_scenes/how_works.rst:23: "When the capture button is hit the driver will
    enter active capturing upon the beginning of the next frame." :30: "When replaying, the initial section of the
    capture (up to the beginning of the frame) is read and executed verbatim." (CS §3)
  - **Vulkan 1.4.368**, chapters/resources.adoc:1098–1100: "The expected usage for this is that a trace capture/replay
    tool will add the VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT flag to all buffers that use
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT" (CS §9)
  - **Tracing:** R. Frostig, M. J. Johnson, C. Leary, "Compiling machine learning programs via high-level tracing",
    SysML 2018: "On a trace cache miss, JAX executes the corresponding Python function and traces its execution into a
    graph of primitive functions with static data dependencies." (RG §B.1)
  - **The display file:** T. H. Myer, I. E. Sutherland, "On the Design of Display Processors", CACM 11(6), 1968, p. 411:
    "The channel takes successive data words from a display file in core until the word count goes zero, at which point
    the central processor restarts the channel at the beginning of the display file." (RG §B.1)
  - **Inspector and executor:** J. Saltz, R. Mirchandaney, K. Crowley, ICASE Report 90-34, 1990: "(1) inspector
    procedures that perform execution time preprocessing and (2) executors or transformed versions of source code loop
    structures. These transformed loop structures carry out the calculations planned in the inspector procedures."
    (RG §B.1)
  - **CUDA:**
    - Runtime API, `cudaStreamBeginCapture`: "When a stream is in capture mode, all operations pushed into the stream
      will not be executed, but will instead be captured into a graph, which will be returned via cudaStreamEndCapture."
      Its `cudaGraphNodeType` lists "GPU kernel node", "Memcpy node", "Memset node" and "External event wait node".
    - Programming Guide v13.5 §4.2.2.1.2.2: "A small number of APIs that enqueue asynchronous operations into streams
      are not currently supported in graphs and will return an error if called with a stream which is being captured".
      §4.2.2.1.2.3: "When an invalid operation is attempted during stream capture, any associated capture graphs are
      invalidated." (CS §4)
  - **Lam 1988, p. 318:** "The steady state of this pipeline constitutes the loop body of the object code." (CP §7)
  - **Metal 4** (macOS SDK 26.4.1; CS §10–12):
    - MTL4ComputeCommandEncoder.h declares 18 copy, fill, mipmap, optimizing and indirect-command-buffer commands
      (:148–562). :323: "Encodes a command that copies data from a buffer instance into another." :456: "Encodes a
      command that fills a buffer with a constant value for each byte."
    - MTL4CommandQueue.h:261: `signalEvent:value:` "Schedules an operation to signal a GPU event with a specific value
      after all GPU work prior to this point is complete." :269: `waitForEvent:value:` "Schedules an operation to wait
      for a GPU event of a specific value before continuing to execute any future GPU work."
    - MTLComputePipeline.h:70–73: `supportIndirectCommandBuffers` "This flag makes this pipeline usable with indirect
      command buffers." MTLDevice.h:834–864 declares six compute-pipeline creation methods.
- **Change:**
  - The capture records every call the captured program makes on the recorder's device, queues and command buffers, as
    apitrace and GFXReconstruct record API calls "to a file". The replay issues the recorded calls in their recorded
    order, as a captured graph holds memcpy and memset nodes beside its kernel nodes. Written at 109b11a3 for copies and
    fills.
  - A function has two captures: of its first execution and of its second. The first is replayed once, for execution
    0, as RenderDoc executes "the initial section of the capture ... verbatim". The second is replayed at every later
    execution, as GFXReconstruct "replays frame `N` again for each iteration", Lam's steady state. A call a library makes
    once, as Dawn's first-use clear, is replayed once.
  - Each call the recorder refuses, or passes on unrecorded, is recorded and replayed in its recorded order by its Metal
    4 counterpart:
    - a blit command by its MTL4ComputeCommandEncoder command;
    - `encodeSignalEvent:value:` and `encodeWaitForEvent:value:` by the queue's `signalEvent:value:` and
      `waitForEvent:value:`, the commands recorded before them committed first ("after all GPU work prior to this
      point is complete"). The signal is still passed to the command buffer during the capture, as now.
  - A call with no Metal 4 counterpart fails the capture, as CUDA invalidates a capture on an unsupported call. It is
    then listed in this text, GFXReconstruct's documented exception.
  - All six of MTLDevice.h's compute-pipeline creation methods create the pipeline from a copy of the program's
    descriptor with `supportIndirectCommandBuffers` set, as GFXReconstruct sets its capture-replay flag on a copy of the
    create info. The three with a completion handler call it with that pipeline.
  - Deleted, at 109b11a3: the refusal of copies, the fill run during the capture, d0a72fc6's blit mode and its direct
    kind.

### 2. A value's copies, the replay's period, and how far ahead execution t starts

- **Code:**
  - mesh 327bea15 and metal-microbench e618ab13: `lead` is the plan's u − 1 in `MetalRecorderRun`'s terms, so
    execution t's requests start once execution t − u has ended. struct mesh_function's `lead` goes, and so do e2b's
    `lead` 1 and mesh_compile.py's 0.
  - `HOST_LAG` is 2 (tools/e2b_replay.m:77; rdma/mesh_compile.py:2128). Before preparing execution t, the host waits
    for execution t − HOST_LAG to end, and writes execution t's inputs into copy t mod HOST_LAG (e2b_replay.m:338–344;
    mesh_compile.py:3918). The host's write is at no level of the compiled program, so ⌈l/s⌉ gives it no q.
  - The period is the least common multiple of every binding's copies, a received operand's phase rows taking 4u
    (metal_recording.m `MetalReplayPeriod`; mesh_compile.py:3297).
  - The phase row of execution t is `(t / u + 1) & 3`. After landing a page, the bridge flips one bit of the page's
    2-bit field, `1 << (field + (round & 1))` (mesh-flow.c:1474).
- **Source:**
  - **M. Lam, PLDI 1988, §2.3, pp. 322–323:**
    - "If the lifetime of a variable is l, and an iteration is initiated every s cycles, then at least ⌈l/s⌉ number of
      values must be kept alive concurrently, in that many locations."
    - "We observe that the minimum degree of unrolling, u, to implement the same schedule is simply max_i q_i. This
      minimum degree of unrolling can be achieved by setting the number of registers allocated to variable v_i to be
      the smallest factor of u that is no smaller than q_i" (RG §B.2; LP §7)
  - **S. S. Bhattacharyya, S. Sriram, E. A. Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", IEEE Trans.
    Signal Processing 45(6), 1997** (the authors' preprint of 11 Feb. 1997; CP §4):
    - "Each processor executes the actors assigned to it in the order specified at compile time. Before firing an
      actor, a processor waits for the data needed by that actor to become available." (RG §B.6)
    - §4.1, p. 9: "Thus if actors v_1, v_2, …, v_n are assigned to the same processor in that order, then G_ipc would
      have a cycle ((v_1, v_2), (v_2, v_3), …, (v_{n – 1}, v_n), (v_n, v_1)), with delay((v_n, v_1)) = 1."
    - p. 10, Eq. (1): "start(v_i, k) ≥ end(v_j, k – delay((v_j, v_i)))".
    - p. 11, Lemma 3: "T = max_{cycle C in G} { Σ_{v is on C} t(v) / Delay(C) }. (4)"
    - §4.3, p. 13: "B_fb(e) = min({Delay(C) | C is a cycle that contains e}) (5)"
  - **R. Reiter, "Scheduling Parallel Computations", JACM 15(4), 1968, p. 595:** "π thus emerges as the minimum possible
    period of any periodic admissible schedule." (CP §6)
  - **design/aot-compilation-change-request.md, "Appendix: the citations the implementation omitted or ignored", item
    2:** "The IPC edges are the plans' transfers, each with C3.3's u copies as its delay".
- **Change:**
  - A value has q = ⌈l/s⌉ copies, l its lifetime and s the interval between executions' starts.
  - Execution t's requests start once execution t − d has ended, d the delay of the transfer's IPC edge, its u copies:
    written at 327bea15 and e618ab13.
  - A host-written input's q:
    - The host and the GPU are two processors of the self-timed schedule. Execution t's input is an IPC edge from the
      host's write to the GPU's reads, of delay 0. The host's wait for execution t − d's end is the edge back, of delay
      d. Each processor's own order is a cycle of delay 1.
    - So by (4), with t_h and t_g the host's and the GPU's times for one execution, T = max(t_h, t_g, (t_h + t_g)/d).
      At d = 1 the cycle through both gives t_h + t_g. At d = 2, (t_h + t_g)/2 ≤ max(t_h, t_g) for every t_h and t_g,
      so no larger d lowers T.
    - By (5) the input's copies are then B_fb = 2. `HOST_LAG` is that q, and its 2 is this derivation's.
  - The replay's period is u = max over the values of q, the host-written inputs' among them. Each value's copies are
    the smallest factor of u no smaller than its q, and the host's wait's delay is its inputs' copies.
  - Where a page's 2-bit field is read (item 3's guard of a page task), the period is 4u, since the field's four values
    recur every four rounds of a copy. It is u otherwise. The least common multiple goes.

### 3. Reading data a remote writer lands: the plan's consumer tasks

- **Code:**
  - `MetalMaskReaders` makes every recorded command that reads a received operand a masked first pass (RG §B.3):
    - each threadgroup compares the phase field of every page of the operand's copy, and returns before its stores if
      any is behind;
    - a compaction lists the threadgroups that returned, as the arguments of a later pass that runs after the queue's
      `waitForEvent:value:` on the waiter's event;
    - both forms are written from the reader's MSL source, and a reader with none raises (1e55e089).
  - The pattern, RG §B.3: presence bits per page in a dense array apart from the data (I-structures' "presence bits"),
    then predication and compaction into an index list for an indirect dispatch, "a topology-driven pass followed by a
    data-driven (worklist) pass", then a coarse wait. RG's verdict on testing the whole operand in every threadgroup:
    "Deviates, not required".
  - Where it came from: Claude's answer to U41 on 2026-11-03, "Masking. [...] lanes or threadgroups whose input pages
    aren't present are predicated off, and that work runs in a later pass when they are" (design/disjoint-pages-
    change-request.md:266–268). Its sources are for predication and compaction in general: the CRAY-1's vector mask,
    SIMT active masks, AVX-512 opmasks, Blelloch's scan (:309–325). None of them is for reading landed data, and none
    says to treat every consumer alike.
  - The recorder takes no consumer kind. `MESH_JOIN` and `MESH_PAGES` are in rdma/mesh-collective.h:55–56,
    tools/e2b_replay.m:609 and rdma/mesh_compile.py:1358, not in metal_recording.{h,m}.
  - The plan's task graph has them:
    - its consumption is "one task for MESH_JOIN and one a page for MESH_PAGES" (mesh-collective.c:1246, :1413);
    - its sets are the ranges "where every declared transfer of the range is MESH_PAGES" (:2360);
    - the transfer schedule is built over that graph: design/plan-to-bridge-change-request.md R4 and R5, the coflows
      and independent flows (PC).
  - e2b's MESH_PAGES consumers (tools/prepare_e2b.py:2040) are R4's function B:
    - `combine`'s TFLite ADDs of the FFN row split's and head split's partial sums (:890);
    - `kv_merge`'s TFLite element-wise operators, the online-softmax merge of the split-KV partials (:1095);
    - the native `maxloc_reduce` (:1461).
    - The first two are recorded LiteRT kernels. They are among the examples of function B in Claude's answer to U71,
      which R4 took up: "elementwise maps, and sum or max reductions (all-reduce and reduce-scatter partials)", "split-K
      GEMM partial sums", "attention by online softmax" (design/plan-to-bridge-change-request.md:296–303).
  - The bridge signals each landing: after landing a page it flips the page's field, then adds one to its copy's count,
    and wakes the count's waiters when it reaches its round's target (mesh-flow.c:1474–1476).
- **Source:**
  - **The operator:** U71 (functions A and B); 2026-11-02 (views of arbitrarily placed pages); U41 (the data edges the
    plan holds and the transport's completions signal; the list in its own section); U48 and U90 (e2b demonstrates
    function B); U192 (the IR the coflow and independent-flow work built, reused for the data processing).
  - **MPI 4.1, §4.2.2, p. 120** (WA §5.3):
    - "The MPI_PARRIVED function can be used to determine if the message data for the indicated partition has been
      received into the receive buffer. Upon success, the receiver becomes free to access the indicated partition (as
      well as any others that previously completed for that operation)."
    - "The completion of a partitioned receive operation through MPI_WAIT or MPI_TEST indicates that the receive buffer
      contains all of the partitions."
  - **Arvind, R. S. Nikhil, K. K. Pingali, "I-Structures: Data Structures for Parallel Computing", TOPLAS 11(4), 1989**
    (the authors' manuscript), p. 13: "the consumer of an array can begin work on parts of the array while the producer
    of the array is still working on other parts." (RG §B.3)
  - **R. Nasre, M. Burtscher, K. Pingali, IPDPS 2013** (WA §1):
    - §II-B, p. 2: "Putting new active nodes onto the worklist can be done by atomically incrementing the length of the
      worklist and then writing an active node's identifier to that location in the worklist."
    - p. 3: "The threads in a thread block can compute the total amount of work to be added by all threads in the
      thread block using a prefix sum (in shared memory) and then use a single atomic instruction per thread block to
      reserve space on the worklist."
    - §III-A, p. 4: "The input and the output worklists are swapped (using pointers) and the accumulated work items in
      the output worklist become the input items for the next iteration."
  - **Metal:** the MSL header metal_command_buffer (GPUCompiler 32023.883) declares `compute_command`, with
    `set_compute_pipeline_state`, `set_kernel_buffer(device T *buffer, uint index)` and
    `concurrent_dispatch_threadgroups(uint3 threadgroups_per_grid, uint3 threads_per_threadgroup)` (:377–423). A kernel
    encodes a dispatch into an indirect command buffer with them.
- **Change:**
  - The GPU's consumers are the plan's consumer tasks, the graph the transfer schedule is built on (U192): a MESH_JOIN
    consumer is one task over its whole operand, and a MESH_PAGES consumer is one task a page. The recorder takes each
    received operand's consumer kind and pages from its declared transfers, the plan's IR, and has no kind of its own.
  - A task is enabled once the pages it reads have landed, as the bridge signals now:
    - a join task once every page of its operand has, its copy's count at its target ("the receive buffer contains all
      of the partitions"; U71: function A is "complete before function A can begin to execute");
    - a page task once its page has, the page's field at its phase ("the receiver becomes free to access the indicated
      partition"; U71: function B "can work on any pages received in any order"; I-structures: "begin work on parts of
      the array while the producer of the array is still working on other parts").
  - The enabled tasks are a worklist: its length and the tasks' identifiers, a dense array in its own section (U41). A
    guard before the consumer's first recorded command reads each task's landings and puts each enabled task on it as
    Nasre et al. state: a prefix sum over a threadgroup, one atomic increment of the length, then the identifiers
    written.
  - The tasks:
    - A join task is the consumer's recorded commands on its whole operand.
    - A page task is the consumer function captured once on a one-page view of its operand (item 1). It is replayed
      with that view bound at each enabled page's place by page-table index (2026-11-02; U166). At e2b's batch of 1, a
      partial is one page (design/plan-to-bridge-change-request.md:1301), so `combine`'s and `kv_merge`'s graphs are
      already a page's.
    - The guard encodes the enabled tasks' dispatches into an indirect command buffer (`compute_command`), each with
      its views bound.
  - The tasks not enabled in the first round are the next round's worklist, run after the queue's wait on the waiter's
    event, where later passes run now ("the accumulated work items in the output worklist become the input items for
    the next iteration"). A task the second round's guard finds not landed is the failure the status records, as the
    later pass's is now.
  - Deleted:
    - the masked first pass, its comparison of every page by every threadgroup, and its compaction of threadgroups;
    - the later pass as the reader with its grid builtins recomputed;
    - the rewriting of the reader's MSL source, and 1e55e089's raise for a reader with none. A task binds views and
      rewrites no source, so a torch MPS kernel can be one.

### 4. Completion: execution t's waits are those of its started requests

- **Code:**
  - rdma/mesh-metal.m `waits_run` walks repetitions 0 to R − 1, with R fixed when the waiter starts. A wait ends with
    ETIMEDOUT after MESH_PLATFORM_NS without progress.
  - R also gives two things item 4 didn't cover. The thread ends after repetition R − 1, and `mesh_metal_waits_end`
    joins it (:307–315). At a cause or after the last wait, the event is set to `final=waiter->repetitions*passes`
    (:218), which releases every GPU wait. e2b passes its `steps` as R (tools/e2b_replay.m:2094–2095).
  - `mesh_transfers_start` appends execution t's submission, and stores the plan's `submissions` with release
    (rdma/mesh-dataflow.c:734–736).
- **Source:**
  - **MPI 4.1 §3.9:**
    - "Often a communication with the same argument list (with the exception of the buffer contents) is repeatedly
      executed within the inner loop of a parallel computation. In such a situation, it may be possible to optimize the
      communication by binding the list of communication arguments to a persistent communication request once and then
      repeatedly using the request to start and complete operations." And: "The request becomes active once the call is
      made."
    - p. 110: "A persistent communication request is deallocated by a call to MPI_REQUEST_FREE (Section 3.7.3). The
      call to MPI_REQUEST_FREE can occur at any point in the program after the persistent request was created. However,
      the request will be deallocated only after it becomes inactive." And: "Create (Start Complete)∗ Free" "where ∗
      indicates zero or more repetitions." (CP §1)
  - **MPI Forum fault tolerance working group, ULFM chapter draft, 9 May 2022** (CP §2):
    - §16.3.1, p. 9: "Once a communicator has been revoked at an MPI process, all subsequent non-local operations on
      that communicator (with the same exceptions as above), are considered local and must complete by raising an error
      of class MPI_ERR_REVOKED at that MPI process."
    - §16.1, p. 1: "any MPI operation that involves a failed process must not block indefinitely but either succeed or
      raise an MPI error".
  - **D. P. Reed, R. K. Kanodia, "Synchronization with Eventcounts and Sequencers", CACM 22(2), 1979, p. 116**
    (RG §B.4; CP §3):
    - "An eventcount can be thought of as a nondecreasing integer variable."
    - "The primitive advance(E) is used to signal the occurrence of an event in the class associated with the
      eventcount E."
    - "we define a primitive await(E, v) that suspends the calling process until the "value" of the eventcount E is at
      least v."
- **Change:**
  - Execution t's waits are the awaits of the requests started for execution t, each awaited once its start has been
    made. The waiter has no count of executions fixed before the first.
  - The waiter takes execution t's waits after an await on the plan's submissions, which `mesh_transfers_start`
    advances: MPI's "Start Complete", once per execution, each wait's bound counted from its start.
  - After execution t's last wait, the event is set to (t + 1) × passes, past t's last pass, as the final value was set
    past the last execution's.
  - The end is the free. `mesh_metal_waits_end` marks the waiter for freeing, and its thread ends once the executions
    started before it have completed ("deallocated only after it becomes inactive"). It is then joined.
  - A failure completes every await, then and later: the event is set to 2^64 − 1, which every await(E, v) meets. The
    error is the waiter's cause and the GPU's status, as now: a revoked communicator's later operations "must complete by
    raising an error".
  - Deleted: `repetitions`, and the final value R × passes.

### 5. The sender's release: every access to the supplied word has the word's size

- **Code:**
  - Before the first implementation, `MetalRelease` issued a seq_cst fence, then `atomic_fetch_add_explicit` on a
    32-bit `atomic_uint` at the low half of the 64-bit supplied word, which the bridge reads with a 64-bit load (RG
    §B.5).
  - metal-microbench 913bdb1f and mesh ee787950: every release is the fence, then `atomic_max_explicit` on the word as a
    device `atomic_ulong`, writing t + 1, with t read from execution t's step word. The kernel's source compiles with
    the Metal toolchain at its default language version and at metal4.0.
- **Source:**
  - **S. Flur et al., "Mixed-Size Concurrency: ARM, POWER, C/C++11, and SC", POPL 2017, p. 2:** "the effective type
    rules of the ISO C standard prohibit mixed-size use of atomics." (RG §B.5)
  - **PTX ISA §8.8:** a release pattern is "a release or acquire-release memory fence followed by a strong write on M in
    program order" (RG §B.5).
  - **design/disjoint-pages-change-request.md C3.7 (:938):** "A supplied word has one writer, its range's producer, which
    stores t + 1 after its last write in execution t."
- **Change:** written at 913bdb1f and ee787950, as above. Every access to the word is 64-bit.

### 6. Placement: pages, with runs only for contiguous-typed functions

- **Code:**
  - rdma/mesh-collective.c `plan_place` places every value page by page, with `mesh_graph_chains` (Dilworth's theorem,
    by Fulkerson's matching) and `mesh_graph_fat_cover` (Mukerji 1994 §3.2) (PC).
  - rdma/mesh-dataflow.c `mesh_view_compose` reserves a range and remaps every typed operand's rows over it with
    `mach_vm_remap`. That comes from design/plan-to-bridge-change-request.md R2 as 4e78f6f4 wrote it: "No value is
    placed as one run." (RG §A).
  - The 11-02 solver placed a contiguous-typed function's values as runs, by TensorFlow Lite's
    `GreedyBySizeAssignment` ported, and every other value page by page (c491c514, f56e59c6, f09d353c, e172a931). It
    was deleted by 0b3e78d7 and is held by tag `pre-return-20261103`.
  - `plan_items` puts a value's blocks in its own item, one a copy, and in the items of the sets that cover them
    (mesh-collective.c:887–906).
  - The bindings take copy c at offset + c × stride (metal_recording.h:76–77,
    `place:operand:in:offset:stride:copies:shares:`).
  - Every e2b value is `run` 1, yet some of e2b's transfers are MESH_PAGES (e2b_replay.m:609).
- **Source:**
  - **The operator:** 2026-11-02 (the typing); U71 (function A's operand "contiguous and complete", function B's
    "complete but not contiguous"); U188 (paging, and the solver for contiguous buffers).
  - **Y. Pisarchyk, J. Lee, arXiv:2001.03288, 2020** (VS; LP §4):
    - §3, p. 2: "Tensor Usage Record of an intermediate tensor t is defined as the triple {first_op_t, last_op_t,
      size_t}, where size_t is t's aligned size in bytes."
    - §5.2: "We first iterate through tensor usage records in non-increasing order by their size (L.1,6). For each
      record, we check already assigned tensors whose usage intervals intersect with that of the current tensor
      (L.10–13) to find the smallest gap in memory between them such that current tensor fits into that gap (L.9,14–17).
      If such a gap is found, the current tensor is allocated to this gap. Otherwise, we allocate it after the rightmost
      tensor whose usage interval intersect with that of the current tensor (L.19–20)."
  - **TensorFlow Lite** (6cf18d88, Apache-2.0), tensorflow/lite/delegates/gpu/common/memory_management.h:130–131:
    "Calculates the assignment of tensors to offsets, considering those tensors are going to be allocated in one
    continuous memory block." (LP §3)
  - **Lam 1988, p. 322:** variable expansion "expands the variable into a higher dimension variable, so that each
    iteration can refer to a different location." (LP §7)
  - **T. Kilburn, D. B. G. Edwards, M. J. Lanigan, F. H. Sumner, "One-level storage system", 1962:** "Hence a system has
    been devised to make the core drum store combination appear to the programmer as a single level of storage, the
    requisite transfers of information taking place automatically." (VS)
  - **design/aot-compilation-change-request.md A8:** "each consumer's kind (R4: a join for function A, a task per page
    for function B)".
- **Change:**
  - A function's operand is one run only where the function is typed `contiguous_input` or `contiguous_output`: U71's
    function A. Function B's operand, "complete but not contiguous", is placed page by page, and so is every other
    function's, as views of its pages by page-table index (U166). e2b's `run` 1 for a value whose consumer is
    MESH_PAGES goes.
  - A set is a range whose every declared transfer is MESH_PAGES, function B's, so no set covers a block of a run. A
    run's blocks are its value's own item.
  - A run's q copies are one usage record of q times its size, Lam's "higher dimension variable". Copy c is at the
    record's offset + c times its size, as the bindings take it.
  - Greedy by Size for Offset Calculation places the runs' records in "one continuous memory block", as f09d353c ported
    TensorFlow Lite's `GreedyBySizeAssignment`. That block is one item of the page placement, live from the runs' first
    use to their last. The chains and the fat cover place it and every other item, as now.
  - Deleted:
    - `mesh_view_compose`'s remapping of typed operands;
    - R2's "No value is placed as one run", for contiguous-typed operands;
    - this item's clause of efa3ad8c, "A run's transfers are consumed whole, as R4's join; every other operand's pages
      are R4's tasks per page". No source states it. A consumer's kind is its declared transfer's (R4).

### 7. Lifetimes: from the first write to the last read

- **Code:**
  - rdma/mesh_compile.py `program()` gave every value of a capture the capture's whole range of levels.
  - metal-microbench 63b9f015 and mesh aca4f783: a captured buffer's interval runs from the level of the first recorded
    command that writes it to the level of the last that reads it.
  - A call's inputs, which the host writes, and its results, which the host reads, keep [0, levels − 1] (aca4f783).
- **Source:**
  - **design/problems-of-14b9a757-change-request.md item 7:** "Every block's item arrives at the first write and departs
    at the last read of the values whose bytes it holds, in the program's order." Its item 14 colours those intervals as
    cyclic intervals (Hendren, Gao, Altman and Mukerji, CC 1992; Mukerji 1994 §3.2).
  - **ExecuTorch** (75af78c4, BSD), exir/memory_planning.py:348–354, `update_tensor_lifetime` (LP §1):
    `if node.op == "placeholder": start = 0`, and every other end is `node_idx if end is None or end < node_idx else
    end`. The output node, whose arguments are the graph's results, is the graph's last: torch/fx/graph.py:1498–1502,
    `output_node = next(iter(reversed(self.nodes)))`.
  - **TensorFlow Lite** (6cf18d88), tensorflow/lite/arena_planner.cc (LP §2):
    - :262–267: "Queue all graph inputs for allocation and make sure they are never overwritten." with `allocate(0,
      tensor_index)`;
    - :334–335: "Note that graph outputs will never be scheduled for deallocation."
  - **B. R. Rau, M. Lee, P. P. Tirumalai, M. S. Schlansker, "Register Allocation for Software Pipelined Loops", PLDI 1992,
    p. 286:** "Live-in values, which are pre-loaded prior to loop entry, must remain in the registers until their
    lifetimes end. This adds a leading blade to the wand of recurring values which extends, on the left, all the way
    through cycle 0." "As discussed, the lifetimes of live-out values, which are used after loop exit, must be extended
    until the loop completes." (LP §5)
- **Change:**
  - A captured buffer's interval runs from the level of the first recorded command that writes it to the level of the
    last that reads it: written at 63b9f015 and aca4f783.
  - A call's input runs from level 0 to the last command that reads it. A call's result runs from the first command
    that writes it to level levels − 1.
  - Only the program's invariants span the whole program.

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
- **Change:** every node runs its captured program by replay, its transfers included, and fcf5d18a's split is deleted.
  Item 3's tasks need no reader source, so in the torch scope this waits on item 3 alone.

## What goes from the earlier texts and their implementation

- The text of 1d4c2ebe, 6ddf657f and 053e4017: its placement sparse buffers, heap argument and single-VM-region point,
  the splitting of the replay at waits, the direct encoding of dispatches and the source-less whole pass. None of them
  cites a method.
- From d0a72fc6:
  - **Kept:** the one received word and one supplied word per range.
  - **Deleted, at 109b11a3 and 1e55e089:** the direct kind, the blit mode, and the whole later pass for a source-less
    reader.
- From mesh 03297e52:
  - **Kept:** the capture at make() and the replay at every execution.
  - **Replaced:** the registration of torch's MPS kernels as the torch scope's kernels, by item 8.
- From efa3ad8c, Claude's text of this request:
  - item 3's "Each consumer row has a count and a dense array of the page indices that have arrived, in its own section
    of the region, in the form of BlockMask's kv_num_block and kv_indices". BlockMask is an attention kernel's mask,
    pre-computed "during compilation time" (WA §4.1), not a source for landed data. Item 3's worklist is Nasre et al.'s,
    of the plan's tasks;
  - item 3's stop for kernels with no MSL source, which its tasks no longer need;
  - item 6's clause on runs' transfers, as item 6 says.

## Stops

- Item 8: which IR carries the torch scope's final lowering.
- Item 1: a call with no Metal 4 counterpart, once one is found. It is listed here when it is.
