# Change request: the capture, reading received data, completion, the release, offsets, lifetimes, the torch scope's final lowering, and every node's replay

Code is cited at mesh aca4f783 and metal-microbench 63b9f015 where a line names no other commit.

The text of each quoted source is recorded in:
- design/replay-grounding-2026-11-06.md;
- design/final-lowering-sources-2026-11-06.md;
- design/capture-replay-sources-2026-11-07.md;
- design/completion-and-period-sources-2026-11-07.md;
- design/worklists-and-arrival-sources-2026-11-07.md;
- design/lifetimes-and-placement-sources-2026-11-07.md;
- design/nccl-sources-2026-11-07.md;
- design/metal-atomic-sources-2026-11-07.md;
- design/tiling-and-bounds-sources-2026-11-07.md;
- design/stops-sources-2026-11-07.md;
- design/list-scheduling-sources-2026-11-08.md.
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

(U41, 2026-11-03 20:35, in part)

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


> "Where each of my phrases came from:" why are any of these being defined at all? why do they need definitions or
> revisions? What is the ambiguity that anyone thought needed resulution?

(U194, 01:21)

> "3. "The recorder never sees the consumer's kind" is also a code fact, " why, broadly, are you trying to 'defien and
> describe code facts'? what does the documentary format of the change request literally do in the user's repeatedly
> communicated system of constraints? what is 'idiomaticity'? why does it matter? these have literala naswers, and
> they have plain meaning wrt each of the numbered phrases as well.
> "  - A page task is the consumer function captured once on a one-page view and bound at each page's place. "
> a page is a region of memory. everything else here is idiomatic and i do not want it described again in the
> immediate session in anything but canonical data structures, algorithms, and citations. thsi si rank, it stinks,
> it's nasy. "At e2b's batch of 1, each partial is already one page." idiomatic, fictional, claude only, lying if
> compared directly against the full history of this transcript in any literal sense. who told oyu taht 'e2b' is
> 'batchsize 1' or has any invariant tensor shape features of that sort? you were not told this by me, or by the
> measuerements and task definitions in the session, or anything that should be cited or construed from. this is the
> idiomaticity that the current owrk is cleraly nad literally and ever emphatically *correcting* by grounding things
> that sound self-evident by weight of repetition with *sources* that *jsutify* their history, evne by saying ' this
> was a clumsy, inarticulate, algorithmically malformed choice that should not survive revision.'
> i have never heard of a 'page task' in my life. i have never heard of a memory indexing scheme where 'instructions'
> use a *minimum* memory size of a 'memory page'.
> "- 4. The waiter. It takes each execution's waits after that execution's start (Reed–Kanodia's await on the plan's
> submission count). I" this is undercontextualized and doesn't make sense. idiomatic; not usnig reference language or
> explanatino of what he data model is or why there is an 'each execution wats after start'. 'waiter' as a noun is
> nonsensical, does not make sense. didn't you say earlier that we have fx graph capture and lowering? idiomaticity?
> " Only function A's operands are runs, and function B's are placed page by page. " idiom idiom idiom idiom. what is
> a 'run'? a 'place'? neither of these are justified by anything.

(U195, 01:33)

> find th4e last instruction term where the user prohibited the use of `plan` as idiomatic. use a subagent; sonnet,
> whatever. there was extensive clarification work on the inappropriate quality of having a polysemantic shorthand of
> a 'plan' whcih is used as a verb, noun, and a reference ofr like 10 different kinds of data structures or inputs,
> outputs, functions, structs, schedulings, allocations?
> this is idiomatic, in the prohibited, unworkable, and 'thing we are removing' sense. i need a better explanatino,
> yet again.

(U196, 01:35)

> ""Batch of 1": no one told me that. It came from my own sentence in plan-to-bridge-change-request.md:1301, which is
> not a source." why did this end up in that document? is there a final citation descirbing a method, algorithm, data
> model, compiler, or runtime from which this must appear? the same should be said for every other detail found so far
> in the past document? why has this *issue* appeared so many times in a row on revisions and implementation passes?
> is there no definition of the change request format in the repository?

(U197, 01:37)

> "The same is true of every detail found so far: " why are there details in the document then? i am going to keep
> using the dialetical method until you articulate what the strictly incorrect work process is taht happens between
> the user issuing plain language authorization to do 'thing a', and instead 'thing a and mysterious thing x, which
> disables thing a' is issued instead?

(U198, 01:41)

> "2. I elaborate A instead of transcribing it. I add sentences that are none of the three kinds the format allows: "
> consider writing a post-commit callback function you can tell yourslf to call which literally imposes a literal code
> review by a literal subagent (wow). as a post commit call can be used evne on committing a change request 'document'
> without implementing it, this solves all problems forever.

(U199, 01:43)

> once the 'change request' document hs been truncated to only the genre of requirement that the
> specification-for-change-requests entails, have an implementatino agent revert all idiomatic edits from the previous
> run, write again whatever must be, etc. i'm rather disappointed by the immediacy and simplicity of the 'do
> absolutely anything but read and use the specification' behavior of the last pass.

(U200, 01:57)

> re point 2: let's treat this as something analogous to the use of cudagraphs in pytorch where we have a foreign
> function api something like a python host can use, compile, and replay for programs invariant up to certain limits
> without recompilation or python-sided or torch-sided overhead thereafter

(2026-09-11 19:29, in part)

> why are we letting torch control any resources (after tracing, compilation, lowering, etc, which should be AOT and
> produce serialized objects we cna resume without needing to repeatedly re-compile anything btw) when *we* control
> *our* own code and calling context?

(2026-10-28 11:45, in part)

> " [...] mesh_compile.py lowers each operator to torch's MPS kernels." did anyone ever say to do this?

(2026-11-06 13:33, in part)

> "- Capture a later execution, after warm-up executions that run outside the capture:" pytorch has aot compile
> through a literal export function. try again, look deeper and wider through the literature before coming back to me
> with 'suggestions' like these.

(2026-11-07 14:43)

> ideally from here we should start to see an explanation of the necessary scope of a lowerer which turns fxgraphs or
> aot outputs into a metal runtime which is complementary to the actual-typing-of-the-linear-algebra rather than
> reproducing errors or misgivings in whatever mps torch backend might already exist. and whether this is a
> complicated and huge task or a trivial feature extension (e.g. naming a replacement pytorch backend which subclasses
> and extensively replaces mediocre behavior within an existing backend; plausibly 500-1500 loc)

(2026-11-07 16:19, in part)

> " Today torch's eager MPS kernels are only captured, never lowered." why would we capture a torch mps kernel when
> torch sucks balls? what was the fxgraph thing again? i'm sort of confused why i keep seeing 'and we also have torch
> eager mps kernels, which are obviously bad...'. what data flow produces them? please explain in less idiomatic
> terms?

(2026-11-07 19:16)

> "6. Recording: these closures run once, inside MetalRecorderCapture, which records the Metal commands torch issues.
> 7. Replay: MetalRecorderReplay re-encodes the commands as Metal 4 indirect command buffers. These replay on every
> execution, bound to the mesh's pages." btw the user almost certainly wrote multiple times that there are more mps
> apis and data types and operations than the indirect command buffer, and that the failure to use the full documented
> api for the scopes of behavior where the api's features are relevant was 'idiomatic' and 'without citation or source
> or ...'. [...] (why capture and replay mps from a bad pointwise lowering of metal when you have the entire fxgraph
> already???) [...] "So matrix multiply became ours, and every other operator stayed torch's eager MPS kernel. Since
> then, no lowering of the FX graph has been written, " is this what the user turns said to do, is what they wanted,
> etc?

(2026-11-07 19:25, in part)

> wrt this specific asked/built gap, which repo types, algorithms, and data structures were implicitly invoked by the
> user or assisatnt turns proximate to that work and not spoken explicitly & w/ citations to the explicit external
> literature? i am only interested in the *citational* failures inherent to this history here as they are the most
> strongly predictive of what goals were recognized and misapplied or misgeneralized in the repo so far (so we can, as
> usual, construct change request documents of the simple and citationally-closed form...)

(2026-11-07 19:34)

> this is an *annoying* but *easy* problem. no huge research or testing-heavy changes are required! the scope of
> impact of these 'change requests' is remarkably small and simple! this is the best case we could ever find from code
> review. [...] "Want me to write them? " sure

(2026-11-07 20:02, in part)

> "Each function's FX graph is lowered the way PyTorch 2 §4.3 states it, one operator at a time:" find a better compiler to cite thanks

(2026-11-07 20:46)

> this is so plainly a self-inflicted harmful choice that we need to walk right back out of it. 
> 1: we hve a data structure of a partially or totally parsed call graph
> 2: all we really have to do is make it into something which is runnable by mps
> 3: mps is not a pointwise eager launch abstract computer. it is actually like, a specific api which has many well written programs it supports.
> 4: we are not here to repeat any and every bad decision torch makes in a default built in compiler. we can read about any compiler we want to and any compiler algorihtms we want to. you iwll notice this already happened for a lot of other topics? like the co-flow independent flow thing? what existingcitations have you simply not remembered orc ontextualized as related to this problem?
> 5:what further citations do they entail?

(2026-11-07 20:48)

The operator, 2026-11-10:

> review the change requests one more time. here are three major kinds of defects to correct:
> 1: misleading pedantic specificity, where there are 23 numbered and lettered subsections which describe something about a source but don't say what algorithm is being used, what data structure is being used, and where it composes in a DAG of functions and the type interfaces between them.
> 2: misleading pedantic vagueness, where terms with no grounding to algorihtms, data structures, or the setting are introduced. `the plan that gates the flimgrundler isn't handled by the srotheclave executor, so we have to stop for that` would be misleading pedantic vagueness, especially with no canonical data structure, algorithm, or method justifying why the change request pertains to `srotheclaves`, `frimgrundlers`, `plans` or `gates`.
> 3: invented qualitative limits that do not come from the change request's plain semantics and plain use as a schema in the first `3` cases of change request documents used in the repo history.

## Requirements

The functions the requirements are stated against are those of the reference program ~/schedule-emit, composed by
`schedule_emit/driver.py` `compile_module`, each from a value to a value:
- `export.export` writes an `ir.Exported`;
- `placement.place` takes its graph and the placement operand, and writes the graph with a `send` and a `recv` node
  for each value read on another processor;
- `devices.on_devices` writes that graph with each node's device;
- `schedule.schedule` takes that graph and the cost operand, and writes one graph per processor, its nodes in
  scheduled order; `schedule.argument_buffers` takes those graphs;
- these are fields of one `ir.Scheduled`, which `lower_inductor.lower`, `runtime.Program` and `archive.save` each
  take.

### 1. Every call is captured and replayed

- **The data structure** is the capture: the program's Metal calls with their arguments, "serialised out in order"
  (RenderDoc), and the contents at the capture's start of each buffer those calls read before writing.
- **The algorithm** is the capture and replay of API calls (apitrace; GFXReconstruct; RenderDoc): at capture each
  call is appended to the capture, and the replay issues the capture's calls in their order.
- This item is placed only in a flow in which the program is executed and recorded before its static schedule exists.
- **Replaces,** in metal-microbench metal_recording.m:
  - the calls refused during a capture: compute-encoder calls not recorded (:3125–3131), blit-encoder calls other than
    `copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:` and `fillBuffer:range:value:` (:3153–3157), and
    command-buffer calls named `encode…` or `…CommandEncoder` (:3211–3214);
  - `encodeSignalEvent:value:`, passed to the command buffer and not recorded (:3183–3184);
  - the compute-pipeline methods of MTLDevice.h that pass the program's descriptor to the device unchanged; three set
    `supportIndirectCommandBuffers` (:2554–2569). A dispatch of a pipeline made without it raises at replay: "Pipeline
    does not support indirect commands" (:3878).
- **Source:**
  - apitrace (3a0ee82a), README: "trace OpenGL, Direct3D, and DirectDraw APIs calls to a file;" and "replay OpenGL and
    Direct3D calls from a file;"
  - GFXReconstruct (f9b0fb0d):
    - README: "The GFXReconstruct project provides tools for the capture and replay of graphics API calls, allowing the
      graphics commands executed by an application to be recorded to a file that may later be replayed to reconstruct
      the graphics-specific behavior of the captured application."
    - docs/GFXReconstruct Architectural Principles.md:89: "If a call truly cannot be represented in its original form
      wholly or piecewise, that case must be documented as an exception." :102: "Some exceptions are necessary in order
      to save information required for replay or modify data so that replay is possible."
    - framework/encode/vulkan_capture_manager.cpp:889–896: "We modify only a copy to prevent the modified pCreateInfo
      from being written to the capture file."
  - RenderDoc (34fbfb43), docs/behind_scenes/how_works.rst:23: "In this state every API call is serialised out
    in order and any initial contents and states are saved."
  - NVIDIA CUDA Programming Guide v13.5: §4.2.2.1.2, "When a stream is being captured, work launched into the
    stream is not enqueued for execution. It is instead appended to an internal graph that is progressively being built
    up." §4.2.2.1.2.3, "When an invalid operation is attempted during stream capture, any associated capture graphs are
    invalidated."
  - Metal 4, macOS SDK 26.4.1:
    - MTL4ComputeCommandEncoder.h:27–29: "Use instances of this abstraction to encode a compute pass into
      ``MTL4CommandBuffer`` instances, as well as commands that copy and modify the underlying memory of various Metal
      resources, and commands that build or refit acceleration structures."
    - MTL4CommandQueue.h:261, `signalEvent:value:`: "Schedules an operation to signal a GPU event with a specific value
      after all GPU work prior to this point is complete." :269, `waitForEvent:value:`: "Schedules an operation to wait
      for a GPU event of a specific value before continuing to execute any future GPU work."
    - MTLComputePipeline.h:70–73, `supportIndirectCommandBuffers`: "This flag makes this pipeline usable with indirect
      command buffers."
- **Write:**
  - Every Metal call of the captured program is recorded, and the replay issues the recorded calls in the order they
    were recorded (apitrace; GFXReconstruct; RenderDoc's "every API call is serialised out in order"), each by the
    call that MTL4ComputeCommandEncoder.h or MTL4CommandQueue.h declares for it.
  - A call for which Metal 4 declares none invalidates the capture (CUDA §4.2.2.1.2.3), and is documented in this
    request (GFXReconstruct :89).
  - Each compute-pipeline creation method of MTLDevice.h (:834–864) creates its pipeline from a copy of the program's
    descriptor, with `supportIndirectCommandBuffers` set (GFXReconstruct :102; vulkan_capture_manager.cpp:889–896;
    MTLComputePipeline.h:70–73).
- **A call the program makes only while a condition holds:**
  - **Replaces:** metal-microbench metal_recording.m at 2f18ac91: `MetalDelegateValues` (:1720–1771) classes a buffer
    the captured calls read before writing as `METAL_INVARIANT`, and `MetalDelegateCopies` (:2749–2786) copies its
    contents once, whether or not the captured calls also write it.
  - **Source:**
    - Dawn (79794682) issues a buffer's clear in `EnsureDataInitialized` only while `NeedsInitialization()`
      holds: `return !mIsDataInitialized && GetDevice()->IsToggleEnabled(Toggle::LazyClearResourceOnFirstUse);`
      (src/dawn/native/Buffer.cpp:1171–1173); `InitializeToZero` then calls `SetInitialized(true);`
      (src/dawn/native/metal/BufferMTL.mm:287–294); the clear is `fillBuffer:range:value:` (:304–306).
    - RenderDoc (34fbfb43), how_works.rst:17, :21, :30 and :38: "In some cases the driver might choose to
      optimise out some of the data-upload calls and lazy initialise the contents of some resources to save on
      background overhead." "In this state every API call is serialised out in order and any initial contents and
      states are saved." "When replaying, the initial section of the capture (up to the beginning of the frame) is read
      and executed verbatim." "When replaying from the beginning of a frame (and not a partial subset of the frame) the
      initial states of all resources are applied, and the initial pipeline state is restored." resource_manager.h:650:
      "Apply the initial contents for the resources that need them, used at the start of a frame".
    - GFXReconstruct (f9b0fb0d), USAGE_desktop_Vulkan.md:1231–1233: "For trimmed capture files, a snapshot of
      the Vulkan API state is written at the start of the file. This state snapshot includes the data for all buffers
      and images that were live at the time that capture started." vulkan_replay_frame_loop_consumer.h:182–183:
      "Tracks a device-local copy of each buffer's contents at the start of the loop range so that GPU writes made to
      the buffer during the loop can be undone before each repetition."
    - CUDA C++ Programming Guide v13.0 §6.2.8.7.5: "A graph is a snapshot of a workflow, including kernels,
      parameters, and dependencies, in order to replay it as rapidly and efficiently as possible."
  - **Write:**
    - The replay of every execution issues such a call where it was recorded: every call is serialised, and the replay
      executes the serialised calls verbatim (RenderDoc :21, :30; CUDA's snapshot). Dawn's clear is a recorded fill,
      replayed at every execution; its condition is not evaluated at replay.
    - The contents at the capture's start of each buffer the captured calls read before writing are saved with the
      capture (RenderDoc :21; GFXReconstruct's snapshot). A buffer the captured calls also write has them applied
      before each execution's replay, so that the execution's writes are undone (RenderDoc :38;
      `ApplyInitialContents`; vulkan_replay_frame_loop_consumer.h:182–183).

### 2. When execution t's requests into one of a value's locations start

- **The data structure** is the node's IPC graph (Bhattacharyya, Sriram and Lee 1997), the CPU and the GPU being two
  of its processors, with the u locations of each value (Lam 1988).
- **The algorithm** is modulo variable expansion (Lam 1988), a start's condition being (1); and, for the number of
  locations of a value the CPU writes, the buffer bound Bff(e) (Bhattacharyya, Sriram and Lee 1997, §4.3) by the
  binary search from (6)'s value, the maximum cycle mean computed at each step.
- **It takes** the IPC graph and each vertex's execution time estimate t(v) ("Source for item 2's execution-time
  estimates", below). **It writes** Bff(e) for each feedforward edge e, which is that value's number of locations.
  - In rdma/mesh-graph.h at mesh 57d9f06e, `mesh_graph_cycle_mean` writes λ_max as a `double`, and `mesh_graph_buffer_bound` writes
    Bff(e) as `uint64_t bound[]`; rdma/mesh-collective.c calls it.
  - The IPC graph is a function of the static schedule: in the reference program, `schedule.argument_buffers` takes
    `schedule.schedule`'s graphs, classes each processor's arguments edge as feedback or feedforward by the same
    paper, and writes `ir.Scheduled.feedforward`, which `runtime.Program` reads.
- **Replaces:** `lead` 1 in tools/e2b_replay.m and 0 in rdma/mesh_compile.py, before metal-microbench e618ab13 and mesh
  327bea15.
- **Source:**
  - M. Lam, "Software Pipelining: An Effective Scheduling Technique for VLIW Machines", PLDI 1988, p. 322:
    "Modulo variable expansion takes advantage of the flexibility of VLIW machines in scalar computation, and reduces
    the number of locations allocated to a variable by reusing the same location in non-overlapping iterations."
  - S. S. Bhattacharyya, S. Sriram, E. A. Lee, "Optimizing Synchronization in Multiprocessor DSP Systems", IEEE Trans.
    Signal Processing 45(6), 1997, preprint p. 10: "start(v_i, k) ≥ end(v_j, k – delay((v_j, v_i))),
    ∀(v_j, v_i) ∈ E_ipc, ∀k > delay(v_j, v_i). (1)"
- **Write:** execution t's requests into one of a value's u locations start once execution t − u's use of that
  location has ended: Lam's reuse of "the same location in non-overlapping iterations", in the form of (1).
- **The number of locations of a value the CPU writes:**
  - **Replaces:** `HOST_LAG` (tools/e2b_replay.m:77; rdma/mesh_compile.py:2128), the constant 2.
  - **Source:**
    - Lam: "If the lifetime of a variable is l, and an iteration is initiated every s cycles, then at least
      ⌈l/s⌉ number of values must be kept alive concurrently, in that many locations."
    - Bhattacharyya, Sriram and Lee 1997, §4.2 and §4.3, pp. 12–14: "If we only have execution time estimates
      available instead of exact values, and we set t(v) in the previous section to be these estimated values, then we
      obtain the estimated iteration period by calculating λ_max." "Since the effect of placing such a restriction
      includes "artificially" constraining src(e) from getting more than Bff(e) invocations ahead of snk(e), its effect
      on the estimated throughput can be modelled by adding the reverse edge d_m(snk(e), src(e)), where
      m = Bff(e) – delay(e), to Gipc". "Bff(e) must be chosen large enough so that the maximum cycle mean remains
      unchanged upon adding d_m(snk(e), src(e))." "An efficient albeit suboptimal procedure to determine Bff is to note
      that if Bff(e) ≥ Σ_{x∈V} t(x) ⁄ λ_max (6) holds for each feedforward edge e, then the maximum cycle mean of the
      resulting graph does not exceed λ_max." "Then, doing a binary search on Bff(e) for each feedforward edge, and
      computing the maximum cycle mean at each search step and ascertaining that it is less than λ_max results in a
      buffer assignment for the feedforward edges." Lemma 3: λ_max is the maximum cycle mean of the IPC graph.
  - **Write:** the CPU's write of a value for execution t is src(e) and the GPU's reads of it snk(e) of a feedforward
    edge e of the node's IPC graph, the CPU and the GPU being two of its processors (each a cycle with one delay,
    Bhattacharyya, Sriram and Lee, preprint p. 9). The value's number of locations is Bff(e): the smallest m for
    which adding d_m(snk(e), src(e)) leaves λ_max unchanged, found by the binary search from (6)'s value, with t(v)
    the execution time estimates. This is Lam's ⌈l/s⌉, l being the execution time along the cycle the reverse edge
    closes and s being λ_max.

### 3. Reading received data

- **Replaces,** in the LiteRT scope: metal_recording.m `MetalReceivedReaders`, `MetalPartitionReady` and `MetalHold`
  at 59b3a3d4 (metal-microbench a8ed7bd6 and 28650030). A recorded dispatch that reads a received operand, or each
  of the parts the partition-ready kernel divides it into, waits at the command queue, before any of its
  threadgroups executes, for every received range that has a page it reads.
- **Source:**
  - the operator: U41 and U71 (the turns);
  - MPI 4.1, §4.2.2, p. 120: "The MPI_PARRIVED function can be used to determine if the message data for the
    indicated partition has been received into the receive buffer. Upon success, the receiver becomes free to access
    the indicated partition (as well as any others that previously completed for that operation)." "The completion of
    a partitioned receive operation through MPI_WAIT or MPI_TEST indicates that the receive buffer contains all of the
    partitions."
  - R. Nasre, M. Burtscher, K. Pingali, "Data-driven versus Topology-driven Irregular Computations on GPUs", IPDPS
    2013: "In topology-driven implementations, all nodes are assumed to be active so the operator is applied
    everywhere even if there is no work to do at some nodes"; "data-driven implementations are organized around a
    worklist".
  - J. Ragan-Kelley et al., Halide, PLDI 2013, §4.2: "For each function, it symbolically evaluates the bounds
    of each dimension based on the bounds required of its caller and the symbolic indices at which the caller invokes
    it."
- **A partition** is MPI_PRECV_INIT's "count: number of elements received per partition" (MPI 4.1 §4.2.1, p. 118):
  the bytes of one page, which metal_recording.m `receive:page:` takes.
- **In the torch scope,** design/torch-source-lowering-change-request.md item 1 writes this item
  (plan-to-bridge-change-request.md R4: "A function that takes pages in any order has one task for each page.").
- **In the LiteRT scope:**
  - **The data structures** are:
    - the presence bits of I-structure storage (Arvind, Nikhil and Pingali 1989), one for each page of a received
      operand: the field its receive's completion writes (design/disjoint-pages-change-request.md C3: "A receive's
      completion is one atomic XOR of one bit of its field"; "A reader of execution t compares the field with
      Gray(⌊t / u⌋ + 1 mod 4). Equal: received.");
    - the mask of a predicated execution (Russell 1978; the RISC-V Vector Extension 1.0), one element for each
      threadgroup of a dispatch;
    - the worklist of a data-driven implementation (Nasre, Burtscher and Pingali 2013): a dense list of threadgroup
      indices, with its length;
    - the task list (Graham 1966) of the dispatches that execute worklists.
  - **The algorithms** are:
    - the index space analysis of Lee (2015, §3.2.2), which classes a kernel as contiguous or discontiguous;
    - stream compaction by a scan and a scatter (Harris, Sengupta and Owens 2007; Blelloch 1990);
    - the ordering of jobs by non-decreasing completion time C̄j (Hall, Schulz, Shmoys and Wein 1997).
  - **It takes** a recorded dispatch, `struct MetalRecordedDispatch` (metal-microbench metal_recording.m:10 at
    59b3a3d4), whose fields are a pipeline, buffers, offsets and a grid; the `source` of the recorded
    `newLibraryWithSource:options:error:` call; the presence bits; and the transfers' completion times, which
    `mesh_plan`'s linear program computes. **It writes** the mask, the worklist, the worklist's length in the
    indirect buffer of the dispatch that executes it, and the task list. The item names no type for the presence
    bits' word, the completion times, the mask, the worklist or the task list.
  - Its input is the recorded dispatch, so in the LiteRT scope this item is placed only in a flow in which the
    program is executed and recorded before its static schedule exists.
  - **Source:**
    - Dawn (79794682), src/dawn/native/metal/ShaderModuleMTL.mm:553–556: `library = AcquireNSPRef([mtlDevice
      newLibraryWithSource:mslSource.Get() options:compileOptions.Get() error:&error]);` MTLDevice.h:786:
      `- (nullable id <MTLLibrary>)newLibraryWithSource:(NSString *)source options:(nullable MTLCompileOptions *)options
      error:(__autoreleasing NSError **)error;` The LiteRT stages' kernels are Tint's MSL, passed there
      (design/lowering-sources-2026-11-06.md).
    - J. Lee, *Virtualizing Data Parallel Systems for Portability, Productivity, and Performance*, Ph.D. dissertation,
      University of Michigan, 2015, Chapter III (SKMD). §3.2, p. 17: "the buffer manager performs static
      analysis on kernels to determine the memory access pattern of each work-group. If the memory access range of each
      work-group can be analyzed statically, the buffer manager will transfer only necessary data back and forth from
      each device once the partitioning decision is made. On the other hand, if the memory access range cannot be
      analyzed, the entire input should be transferred to each device". §3.2.2, p. 25: "For index space analysis, the
      buffer manager uses data flow analysis focusing on the index operand of store and load instructions [...]. Using
      use-def chains, the buffer manager computes the function of index. If the function is affine and represented as
      a · (W0 · w0 + l0), it is defined as a Contiguous Kernel. In this equation, a is a constant or an induction
      variable of loop, and wi, li, and Wi represents work-group ID, work-item ID, and size of work-group in the i-th
      dimension, respectively. For this type of kernel, it is safe to transfer a subset of data to each device
      proportional to assigned work-groups." "if the index space of the kernel cannot be determined statically, or the
      affine function fails to be recognized as above, the buffer manager gives up optimizing data transfer and
      defines it as a discontiguous kernel. In this case, the entire input and output will be transferred back and
      forth between the host and external devices".
    - MPI 4.1 §4.2.3, p. 121: "Partitions are contiguous in memory, there is no padding in between them."
    - Predication: R. M. Russell, "The CRAY-1 Computer System", CACM 21(1), 1978: "the 64-bit vector mask
      (VM) register controls vector element designation in vector merge and test instructions. Each bit of the VM
      register corresponds to a vector register element." The RISC-V Vector Extension 1.0: "The mask value used to
      control execution of a masked vector instruction is always supplied by vector register `v0`." Apple, Metal
      Shading Language Specification 4.1, §6.10.2, `simd_ballot`: "The function sets the bits that correspond to
      inactive threads to 0." §6.10.1, p. 216: "If threadgroup_barrier (or simdgroup_barrier) is inside a conditional
      statement and if any thread enters the conditional statement and executes the barrier function, then all
      threads in the threadgroup (or SIMD-group) need to enter the conditional and execute the barrier function."
    - Stream compaction: M. Harris, S. Sengupta and J. D. Owens, "Parallel Prefix Sum (Scan) with CUDA",
      GPU Gems 3, ch. 39, 2007: "Informally, stream compaction is a filtering operation: from an input vector, it
      selects a subset of this vector and packs that subset into a dense output vector." "Stream compaction requires
      two steps, a scan and a scatter." G. E. Blelloch, "Prefix Sums and Their Applications", 1990, p. 46: "The basic
      idea is to determine a new index for each element and then permute the elements to these new indices using an
      exclusive write."
    - The worklist: Nasre, Burtscher and Pingali, above, and p. 3: "The threads in a thread block can
      compute the total amount of work to be added by all threads in the thread block using a prefix sum (in shared
      memory)".
    - The presence of each element: Arvind, R. S. Nikhil and K. K. Pingali, "I-Structures: Data Structures
      for Parallel Computing", TOPLAS 11(4), 1989, pp. 12–13: "a consumer that tries to read some A[i] while it is
      still empty is made to wait until the corresponding (f i) has completed. One way to achieve this synchronization
      is to use I-structure storage, where each location has "presence bits" to indicate whether the value is present
      or absent." p. 13: "the consumer of an array can begin work on parts of the array while the producer of the
      array is still working on other parts."
    - The indirect dispatch: Apple, MTL4ComputeCommandEncoder.h:90–112, macOS SDK 26.4.1: "Encodes a
      compute dispatch command with a grid that aligns to threadgroup boundaries, using an indirect buffer for
      arguments." "This method allows you to supply the threadgroups-per-grid counts indirectly via an ``MTLBuffer``
      index. This enables you to calculate this value in the GPU timeline from a shader function, enabling GPU-driven
      workflows."
    - The order of the dispatches that execute worklists: L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res.
      22(3), 1997, §2.1, p. 5 of the preprint: "first, we obtain an optimal solution to the linear program,
      C̄1, ..., C̄n; and then we schedule the jobs in order of non-decreasing C̄j, where ties are broken by choosing an
      order that is consistent with the precedence relation." R. L. Graham, Bell System Technical Journal 45(9), 1966,
      §1.1, p. 1564: "We are given a linear ordering L: (T_k1, ···, T_km) of T called a task list (or priority
      list)."
  - **Write:**
    - A recorded kernel's source is the `source` of the recorded `newLibraryWithSource:options:error:` call (item 1
      records every call). The index function of each of its loads and stores is computed from that source by
      use-def chains; a kernel whose index functions are affine, a · (W0 · w0 + l0), is a contiguous kernel, and any
      other is discontiguous (Lee §3.2.2).
    - The pages a threadgroup of a contiguous kernel reads are the partitions its index functions' values over its
      threads fall in ("Partitions are contiguous in memory"). A threadgroup of a discontiguous kernel reads every
      partition, as Lee transfers "the entire input" for one.
    - Each recorded dispatch that reads a received operand is first executed predicated by the mask (Russell; RISC-V
      V). Each threadgroup tests the presence bits of the pages it reads (I-structures' "presence bits"), and no
      other. A threadgroup any of whose presence bits shows its page absent returns before any store and writes 1 to
      its element of the mask. It passes or returns as a unit, its threads' results joined by `simd_ballot` (MSL
      §6.10.1 and §6.10.2).
    - A stream compaction after that dispatch packs the indices of the threadgroups whose mask element is 1 into a
      dense list by a scan and a scatter, and writes the list's length as the threadgroup count of the dispatch that
      executes the list (Harris, Sengupta and Owens; Blelloch). The list is the worklist of a data-driven
      implementation (Nasre, Burtscher and Pingali).
    - A second dispatch of the kernel executes the worklist's threadgroups, its threadgroup count read from an
      indirect buffer (MTL4ComputeCommandEncoder.h:90–112). Each of its threadgroups tests its presence bits again.
    - The dispatches that execute worklists are a task list (Graham) in the order of non-decreasing completion times
      of the transfers they read, the times `mesh_plan`'s linear program computes, "where ties are broken by
      choosing an order that is consistent with the precedence relation" (Hall et al.).
    - No source quoted here says how a dispatch that executes a worklist is ordered against the recorded calls that
      read what it writes.

### 4. The completion of execution t's requests

- **The data structures** are the persistent communication request of MPI 4.1 §3.9, whose life is "Create (Start
  Complete)∗ Free", and the eventcount of Reed and Kanodia (1979), "a nondecreasing integer variable": the Metal
  event's `uint64_t` value (MTLEvent.h:64, `signaledValue`).
- **The algorithms** are await(E, v) (Reed and Kanodia 1979) and, at a failure, `complete_all` (Linux v6.12,
  kernel/sched/completion.c).
- **It takes** each execution's start, which `mesh_transfers_start` (rdma/mesh-dataflow.c:734–736) writes as its
  store of `submissions`, a field of `struct mesh_state`. **It writes** the event's value, which the command queue's `waitForEvent:value:` reads.
- **Replaces:** rdma/mesh-metal.m `waits_run`, which walks executions 0 to R − 1 and then sets the event that the
  command queue's `waitForEvent:value:` waits on to R × `passes` (:218, :222). `mesh_metal_waits_start` takes R as
  `repetitions` (:258), and e2b passes its `steps` (tools/e2b_replay.m:2094–2095).
- **Source:**
  - MPI 4.1, §3.9: "The request becomes active once the call is made." p. 110: "A persistent
    communication request is deallocated by a call to MPI_REQUEST_FREE (Section 3.7.3). The call to MPI_REQUEST_FREE
    can occur at any point in the program after the persistent request was created. However, the request will be
    deallocated only after it becomes inactive." And: "Create (Start Complete)∗ Free" "where ∗ indicates zero or more
    repetitions."
  - D. P. Reed, R. K. Kanodia, "Synchronization with Eventcounts and Sequencers", CACM 22(2), 1979, p. 116: "we
    define a primitive await(E, v) that suspends the calling process until the" … "of the eventcount E is at least v."
  - MPI Forum fault tolerance working group, ULFM chapter draft, 9 May 2022, §16.3.1, p. 9: "Once a communicator
    has been revoked at an MPI process, all subsequent non-local operations on that communicator (with the same
    exceptions as above), are considered local and must complete by raising an error of class MPI_ERR_REVOKED at that
    MPI process."
- **Write:**
  - Execution t's waits are those of the requests started for execution t, each awaited once its start is made ("The
    request becomes active once the call is made"), for any number of executions ("zero or more repetitions"). No count
    of executions is fixed before the first. A start is `mesh_transfers_start`'s store of `submissions`
    (rdma/mesh-dataflow.c:734–736).
  - The waits end at the free, once the requests started before it are complete ("deallocated only after it becomes
    inactive").
- **The value the event is set to at a failure:**
  - **Replaces:** `waits_run`'s R × passes (rdma/mesh-metal.m:218, :222) and `repetitions` (:258).
  - **Source:**
    - Linux v6.12, kernel/sched/completion.c:52–78: "complete_all: - signals all threads waiting on this
      completion" "This will wake up all threads waiting on this particular completion event." "Since complete_all()
      sets the completion of @x permanently to done to allow multiple waiters to finish, a call to reinit_completion()
      must be used on @x if @x is to be used again." `x->done = UINT_MAX;` Documentation/scheduler/completion.rst:252:
      "or calls complete_all() to signal all current and future waiters".
    - Reed and Kanodia: "An eventcount can be thought of as a nondecreasing integer variable." await(E, v)
      "suspends the calling process until the "value" of the eventcount E is at least v."
    - Metal, macOS 26.4 SDK: MTL4CommandQueue.h:266–267 `signalEvent:(id<MTLEvent>)event value:(uint64_t)value`;
      :274–275 `waitForEvent:(id<MTLEvent>)event value:(uint64_t)value`; MTLEvent.h:64 `uint64_t signaledValue`. Apple's
      MTLSharedEvent documentation: "When you set the value of a shared event, its value is changed only if you
      provide a larger value than the value currently stored in the event."
    - ULFM §16.3.1: after a revoke, operations "must complete by raising an error of class MPI_ERR_REVOKED";
      Bland et al. 2013 §5.4: "After a communication object has been revoked, it can never be repaired."
  - **Write:** at a failure the event is set to UINT64_MAX, the greatest value of its `uint64_t`, as `complete_all`
    sets `done` to UINT_MAX: every wait on it, present and future, for any value, is satisfied (await(E, v) returns
    once E ≥ v), with no count of executions. The CPU's waits raise the error (ULFM §16.3.1). The event is not
    signalled again (`reinit_completion` is `complete_all`'s condition for reuse; a revoked object "can never be
    repaired"); the next view's requests take a new event.

### 5. The release writes its word in one 64-bit access

- **The data structure** is one 64-bit atomic word, `atomic_ulong` (metal_atomic:104).
- **The algorithm** is the release pattern of PTX ISA §8.8, "a release or acquire-release memory fence followed by a
  strong write on M in program order", the write being `atomic_max_explicit`.
- **It takes** the execution's number t and **writes** the word as t + 1. The kernel source that `MetalRelease`
  (metal-microbench metal_recording.m) returns writes it; the bridge reads it (design/disjoint-pages-change-request.md
  C3: "The bridge posts a send of execution t once the word reads at least t + 1.").
- **Replaces:** at metal-microbench 0771d07a, `MetalRelease`: a seq_cst fence, then `atomic_fetch_add_explicit` on a
  32-bit `atomic_uint` at the low half of a 64-bit word, whose note gives the word t + 1 after execution t's release.
- **Source:** S. Flur et al., "Mixed-Size Concurrency: ARM, POWER, C/C++11, and SC", POPL 2017, p. 2: "the
  effective type rules of the ISO C standard prohibit mixed-size use of atomics." PTX ISA §8.8: a release
  pattern is "a release or acquire-release memory fence followed by a strong write on M in program order". Metal's
  metal_atomic header, GPUCompiler 32023.883: `typedef _atomic<ulong> atomic_ulong;` (:104); the types an atomic
  store takes are `device int *`, `threadgroup int *`, `device uint *` and `threadgroup uint *` (:168–171); the one an
  atomic max takes is `device ulong *` (:641).
- **Write:** the fence, then one 64-bit atomic write of the word, `atomic_max_explicit` with t + 1, the 64-bit atomic
  write the header declares for it.

### 6. Offsets for the operands of functions typed contiguous

- **The data structure** is the tensor usage record of Pisarchyk and Lee (2020): a size and a usage interval
  (`first_task`, `last_task` in TensorFlow Lite). The result is one offset for each record in one memory block, and
  the block's `total_size`.
- **The algorithm** is Greedy by Size for Offset Calculation (Pisarchyk and Lee 2020, §5.2), as TensorFlow Lite's
  `GreedyBySizeAssignment` states it, its intersection test being `do_arcs_overlap` on the ring of the u executions.
- **It takes** one usage record for each of the u locations (item 2) of each operand of a function typed
  `contiguous_input` or `contiguous_output`, the record's interval being that location's lifetime. **It
  writes** each record's offset and `total_size`. The interval is over the u executions of the unrolled steady
  state, which is the static schedule's (Lam 1988), so the records are a function of `schedule.schedule`'s result.
  The item names no function that writes the offsets and none that reads them, and no type for an offset.
- **Replaces:** rdma/mesh-dataflow.c `mesh_view_compose` (:187–208 at mesh f55c541c), which reserves an address range
  and remaps operands' page-table rows over it with `mach_vm_remap`.
- **Source:**
  - the operator: 2026-11-02 and U188 (the turns);
  - Y. Pisarchyk, J. Lee, "Efficient Memory Management for Deep Neural Net Inference", arXiv:2001.03288, 2020:
    §5, "We call the other memory sharing approach Offset Calculation where a large chunk of memory is pre-allocated and
    the intermediate tensors are given parts of the memory by the offsets within the memory block." §5.2, "We first
    iterate through tensor usage records in non-increasing order by their size (L.1,6). For each record, we check
    already assigned tensors whose usage intervals intersect with that of the current tensor (L.10–13) to find the
    smallest gap in memory between them such that current tensor fits into that gap (L.9,14–17)."
  - TensorFlow Lite (6cf18d88), tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.cc:
    "If no suitable gap found, we should allocate current tensor after the rightmost tensor, which usage
    interval intersects with the current one."
  - M. Lam, PLDI 1988, p. 322: variable expansion "expands the variable into a higher dimension variable, so that
    each iteration can refer to a different location."
- **Write:**
  - Each operand of a function typed `contiguous_input` or `contiguous_output` is given an offset in one memory block by
    Greedy by Size for Offset Calculation, as TensorFlow Lite's `GreedyBySizeAssignment` states it.
  - `mesh_view_compose`'s remapping of those operands is deleted.
- **Where the block lies, and the usage records of the u locations:**
  - **Source:**
    - TensorFlow Lite (6cf18d88), arena_planner.h:141–149: "Raw memory buffer that is allocated for all
      temporary and graph outputs that are declared kTfLiteArenaRw." `SimpleMemoryArena arena_;` "Raw memory buffer
      that is allocated for persistent tensors that are declared as kTfLiteArenaRwPersistent."
      `SimpleMemoryArena persistent_arena_;` arena_planner.cc:433–436: `arena_.Commit(&arena_reallocated)` and
      `persistent_arena_.Commit(&persistent_arena_reallocated)`, each its own. simple_memory_arena.cc:441–445: "Resize
      the arena to the high water mark (calculated by Allocate), retaining old contents and alignment in the process.
      Since Alloc pointers are offset based, they will remain valid in the new memory block."
      memory_management.h:130–131: "Calculates the assignment of tensors to offsets, considering those tensors
      are going to be allocated in one continuous memory block."
    - Hendren, Gao, Altman and Mukerji, ACAPS Memo 33, §2.2, p. 5 (quoted in design/problems-of-14b9a757-change-request.md
      item 14): "each variable in the original loop will be split into several variables depending the unrolling
      factor. Then, the interval graph can be constructed where each new variable in the unrolled loop is associated
      with a cyclic interval."
    - Mukerji 1994, §3.1, p. 29 (the same item): "Cyclic intervals wrap around from one iteration to the next, it is
      one long interval. However, in the interval graphs, we choose to depict it as two intervals instead."
    - Dagor Engine daFrameGraph, heapLayout.h:19 and heapLayout.cpp:13–23 (the same item): "An arc of the frame ring
      whose end is not after its start wraps the ring"; `do_arcs_overlap`, quoted there in full.
    - TensorFlow Lite greedy_by_size_assignment.cc:93–97: the test that skips an allocated record,
      `usage_records[allocated_id].last_task < rec->first_task || usage_records[allocated_id].first_task >
      rec->last_task`.
  - **Write:**
    - The memory block is its own allocation, of Greedy by Size's `total_size`, committed apart from every other
      operand's pages as `arena_` and `persistent_arena_` are committed apart: consecutive pages, in which no page
      of another operand lies. Nothing else constrains where it lies, the other operands being "arbitrary views of
      arbitrarily placed pages" (2026-11-02).
    - An operand's u locations are u usage records, one per location, each with the operand's size and the interval
      of that location's lifetime over the u executions of the unrolled steady state (Memo 33 §2.2). A record whose
      interval wraps from the last execution into the first is one interval depicted as two (Mukerji §3.1); the
      intersection test of greedy_by_size_assignment.cc:93–94 is `do_arcs_overlap` on the ring of the u executions, and
      nothing else of `GreedyBySizeAssignment` changes.

### 7. Lifetimes

- **The data structure** is the lifetime of Lam (1988), "the duration between the first assignment into the variable
  and its last use": for each buffer, an interval of positions in the capture's commands.
- **The algorithm** is the lifetime computation of ExecuTorch's exir/memory_planning.py and TensorFlow Lite's
  arena_planner.cc: a buffer's interval runs from its first write to its last read, an input's starts at the first
  position, and a result's ends at the last.
- **It writes** one interval for each buffer. The item names no type for it and no function that reads it.
- The positions are those of the capture's commands, so this item is placed only in a flow in which the program is
  executed and recorded before its static schedule exists.
- **Replaces:** rdma/mesh_compile.py `program()`, which gives a call's inputs and its results the capture's whole range
  of levels, [0, levels − 1] (aca4f783).
- **Source:**
  - M. Lam, PLDI 1988, p. 322: "The lifetime of a register variable is defined as the duration between the first
    assignment into the variable and its last use."
  - ExecuTorch (75af78c4), exir/memory_planning.py:348–354: `if node.op == "placeholder": start = 0`; and
    torch/fx/graph.py:1498–1502: `output_node = next(iter(reversed(self.nodes)))`.
  - TensorFlow Lite (6cf18d88), tensorflow/lite/arena_planner.cc: "Queue all graph inputs for allocation and
    make sure they are never overwritten." :334–335: "Note that graph outputs will never be scheduled for deallocation."
  - B. R. Rau, M. Lee, P. P. Tirumalai, M. S. Schlansker, "Register Allocation for Software Pipelined Loops", PLDI 1992,
    p. 286: "Live-in values, which are pre-loaded prior to loop entry, must remain in the registers until their
    lifetimes end. This adds a leading blade to the wand of recurring values which extends, on the left, all the way
    through cycle 0." "As discussed, the lifetimes of live-out values, which are used after loop exit, must be extended
    until the loop completes."
- **Write:**
  - A captured buffer's interval runs from its first write to its last read.
  - A call's input's interval starts at the capture's first command, as a placeholder's `start = 0` (ExecuTorch), and
    ends at its last read. A call's result's interval starts at its first write and ends at the capture's last
    command, the output node being the graph's last (torch.fx) and a graph output "never ... scheduled for
    deallocation" (TensorFlow Lite).

### 8. The torch scope's final lowering

- **Replaces:** the torch scope's kernels, which are torch's MPS kernels, one ATen operator at a time (mesh 03297e52).
- **Source:** read and quoted in design/final-lowering-sources-2026-11-06.md.
  - J. Ansel et al., PyTorch 2, ASPLOS 2024, §1: "The framework only has visibility of a single operator at a time, and
    thus cannot automatically perform optimizations, like fusion or scheduling, that cross operator boundaries."
  - T. Chen et al., TVM, OSDI 2018, abstract: "TVM solves optimization challenges specific to deep learning, such as
    high-level operator fusion, mapping to arbitrary hardware primitives, and memory latency hiding."
  - J. Ragan-Kelley et al., Halide, PLDI 2013, §1: "decoupling the algorithm definition from its execution strategy".
  - P. Tillet, H. T. Kung, D. Cox, Triton, MAPL 2019, §5.2.1: "Nested tiling strategies … aim at decomposing tiles into
    micro-tiles and eventually nano-tiles in order to fit a machine's compute capabilities and memory hierarchy as
    tightly as possible."
  - B. Spector et al., ThunderKittens (arXiv:2410.20399), §1: "any high performance framework must prioritize keeping
    tensor cores at high utilization whenever possible."
  - C. Cecka, CuTe layout algebra (arXiv:2603.02298), abstract: "These instructions prescribe fixed, often complex data
    layouts that must be correctly propagated through the entire execution pipeline to ensure both correctness and
    optimal performance."
  - T. Dao et al., FlashAttention, NeurIPS 2022, abstract: "an IO-aware exact attention algorithm that uses tiling to
    reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM."
  - Y. Xu et al., GSPMD (arXiv:2105.04663), §1: "GSPMD will run compiler passes that complete the sharding specification
    on the entire computation graph, and transform it into a mathematically equivalent, parallelized computation to run
    on each device."
  - N. Shazeer et al., Mesh-TensorFlow, NeurIPS 2018, abstract: "A Mesh-TensorFlow graph compiles into a SPMD program
    consisting of parallel operations coupled with collective communication primitives such as Allreduce."
  - L.-W. Chang et al., FLUX (arXiv:2406.06858), abstract: "Flux overdecomposes communication and computation operations
    into much finer-grained operations and further fuses them into a larger kernel to effectively hide communication
    without compromising kernel efficiency."
  - S. Zheng et al., TileLink (arXiv:2503.20313), abstract: "TileLink decouples the design space of communication and
    computation, linking these two parts via tile-centric primitives."
  - A. Jangda et al., CoCoNet, ASPLOS 2022, §1: "we need to schedule the MatMul kernel to compute chunks in the order the
    AllReduce kernel communicates them."
  - Triton-distributed (arXiv:2504.19442), §2.1: "The core of our programming model includes three concepts: symmetric
    memory, signal exchange, and async-task."
  - Metal Performance Primitives, MPPTensorOpsMatMul2d.h:216–218 (macOS 26.4 SDK): "cooperative_tensor owns thread
    private data and divides the data for entire tensor among threads (participating in the scope of operation) in
    implementation defined manner."
- **Written** in design/torch-source-lowering-change-request.md, items 1–3.

### 9. Every node replays

- This item is placed only in a flow in which each node's program is executed and recorded before its static schedule
  exists.
- **Replaces:** mesh fcf5d18a: a node where a declared transfer names one of its values calls torch from Python at every
  execution, while the other nodes replay.
- **Source:** NCCL User Guide 2.32.3: "Collective operations have to be called for each rank (hence CUDA device),
  using the same count and the same datatype, to form a complete collective operation." "Starting with NCCL 2.9, NCCL
  operations can be captured by CUDA Graphs."
- **Write:** every node runs its captured program by replay, its collectives included. fcf5d18a's calls of torch from
  Python at every execution are deleted.

## Source for item 2's execution-time estimates

- **The data structure** is one execution time estimate t(v) for each compiled function, a number.
- **The algorithm** is the analytical cost model of XLA's GPU performance model, "Device time = max(compute time,
  memory read+write time) + kernel launch overhead", with the machine's rates obtained as Roofline obtains them,
  "once per multicore computer, not once per kernel".
- **It takes** each compiled function's operations, the bytes it reads and writes and its execution mapping, which
  the target lowering supplies, and the machine's rates and primitive costs as operands. **It writes** t(v), which
  item 2's computation of λ_max and Bff(e) takes. In rdma/mesh-graph.h at mesh 57d9f06e, `mesh_graph_actor_time` takes a `struct
  mesh_graph_machine` and a `struct mesh_graph_actor` and writes the time as a `double`.
- **Applies to:** item 2's t(v), and aot-compilation-change-request.md A3's weighting operand.
- **Replaces:** the absence of a producer for t(v), and the consequent constant `HOST_LAG`.
- **Source:**
  - OpenXLA authors, [Cost Models in XLA GPU — Present and Future](https://github.com/openxla/xla/discussions/10065),
    “Status-quo”: “Device time = max(compute time, memory read+write time) + kernel launch overhead”.
    Its compute estimate uses operation costs and attainable parallelism; memory estimates use the accessed bytes
    and the target's bandwidth. The corresponding
    [GpuPerformanceModelBase](https://github.com/openxla/xla/blob/main/xla/service/gpu/model/gpu_performance_model_base.cc)
    methods are `ComputeTime`, `ReadTimeWithDRAMHeuristic`, `WriteTime`, and
    `CombineComputeAndMemoryAccessTime` (read 2026-11-09).
  - S. Williams, A. Waterman and D. Patterson, [Roofline](https://escholarship.org/content/qt78h8v7mr/qt78h8v7mr.pdf),
    §3–4: “these limits are created once per multicore computer, not once per kernel.” The model uses processor and
    memory rates, with ceilings for the parallelism and instruction mix attained by the implementation; §3 and
    Appendix A obtain those machine operands from hardware specifications or measurements.
  - Bhattacharyya, Sriram and Lee §4.2–4.3 and Lam, as quoted in item 2, supply the consumer of these estimates.
  - Bhattacharyya, Sriram and Lee, §1: they "assume that "good" estimates are available for the execution times of
    actors".
- **Write:** the target lowering supplies the operations, memory traffic and execution mapping of each compiled
  function to the analytical cost model, with the machine's rates and primitive costs as operands. CPU work,
  including the CPU's write of item 2, uses its CPU rates and traffic; a GPU function uses its GPU mapping and
  rates. The estimates are stored with the compiled functions and are the t(v) of item 2's λ_max and Bff(e). The
  resulting Bff(e) supplies the value's locations and the native program's corresponding reuse edges.
  Transformations update the costs of their resulting functions along with their effects.
