# Change request: the synchronization signature and its monitor

Code is cited at mesh e8cd57e0 and metal-microbench 21c44310.

## The turns

> "- The aot and plan-to-bridge "What judges it" sections fall outside the grammar. They are unsourced assertions." are these inside of change request documents or are they other materials?
> "- A sourced measurement requirement was allowed all along, and nobody wrote one." the first three change request documents didn't ask for measurements and were written to be tamper-evident to catch and prevent imaginary or agent-invented 'judgements' or 'measurements' for a problem rleated to the functional arragnement of code, the repository's data structures and algorithms, etc. the change request documentary format was *concretely used* to catch defects-by-extrapolation in those first 3 cases; there are still no explicit perfomrance requirements in any user directed commentary on the change requests because here, in this sitautoin, if the citations used in the uesr-assistant dialogue had been used strictly, the syntactic limits on how code is parsed and lowered would have produced runtimes that do not have tons of command buffer launch overheads or spurious waits, guards, syncs, or serializations.

(2026-11-08)

**Claude:**

> (・_・;) They're inside the change request documents, so I was wrong to put them outside the grammar.
>
> **Where they are:**
> - **Every early request has one,** inside "Claude's proposal": page-queue ("Order, and what judges it"), disjoint-pages (proposal item 5, then again in the revisions in appendices C and D), plan-to-bridge and aot-compilation. Each was there from that document's first commit.
> - **The proposal is part of the document:** T178 made it so ("the unit of work *is* the change request document once the claude proposal is part of it").
> - **They check how the code is arranged, not how fast it runs:**
>   - "The bridge is given pages, bytes, words and which entries are a set's, and nothing else."
>   - "No kernel is written by the compilation."
>   - "Every cable's sends follow the schedule of R5's method."
> - **That is the tamper-evidence you describe:** a stated arrangement that any addition visibly breaks.
>
> **On measurements, you're right, and I retract "nobody wrote one".** Nothing was missing. The requests left measurement out on purpose, and none was needed. The sources quoted in the dialogue already rule the overheads out:
> - **CUDA's graph,** quoted in torch-capture item 1: "a snapshot of a workflow, including kernels, parameters, and dependencies, in order to replay it as rapidly and efficiently as possible."
> - **Lee and Ha's self-timed strategy:** "Each processor executes the actors assigned to it in the order specified at compile time." Processors synchronize only where data crosses between them.
> - **Bhattacharyya, Sriram and Lee (1997):** the paper is about removing synchronizations, not adding them.
> - **I-structures:** "the consumer of an array can begin work on parts of the array while the producer of the array is still working on other parts."
> - **C3.3:** "At run time there are no allocations and no bookkeeping; every address is a constant in the plan."
> - **Your turn in R4:** "who says any part of this code is synchronized s.t. there is a global sync point which interrupts numeric backends each and every time any page has been received from dma?"
>
> None of these contains a commit and a queue wait before every later pass, so the regression is entirely departures from the citations: mine first, in request text such as item 3's queue wait, then the code's. The review now running is the matching check, because it reads each commit against the sources' own statements. I'll report what it marks outstanding when it finishes.

> "The sources quoted in the dialogue already rule the overheads out:" so we need to deduce what implementing agents have been doing besides refactoring, simplifying, and centarlizing the source of the rpoject upon its rqeuirements to consume a callgraph, fxgraph parse and lower it through a lowerer, ensuring the lowerer has certain properties, then run the lowered code (with no other entry points to the collective opwerations we measure than this singular simple path); using the optimizing compiler features described in the citations to *choose* lowerings which are compatible iwth all of the goals we have, to *implement* lowerings which are compatibel iwth the properties of the dma driver and its necessary wrapper code, compatible iwth the properties of linear algebra and the dependency structures we know ahead of time about linear algebraic functions which are executed in particular orders in particular callgraphs. what do we feel about recurrence here? well, like cudagraphs adn friends we implicitly took this position that we only parse or handle directed graphs with finite numbers of (directed) cycles, s.t. we can explicitly unroll all cycles to bigger directed acyclic graphs with very convenient and obvious and compiler-IR-legible type annotations like 'x_17 is the same parameters but a different operand from x_3 in the 18-deep weight-tied depthwise-recurrent RESNET_18 we're using'. (RESNET_18 is a purely hypothetical architecture, you're not in a kaiming he jumpscare horror format here).
>
> "None of these contains a commit and a queue wait before every later pass, "
> consider highreflectinguinely that claudes have an unconditional behavior of always trying to add queue waits and allsyncs and serializations to partially optimized (or even fully optimized) async networking code, and that all of the documentary structure of the chagne request format and similar is here to mitigate and channel claudes away from unconditional behavior.

(2026-11-08)

**Claude:**

> (－－;) I can see it in the code, and it's the behavior you describe: the agents' work, and my change requests, turned an unrolled replay into a host-stepped, synchronized one.
>
> **First, a correction to the starting point.** The 239 tok/s I gave you came from a program that no longer exists: mesh_rank.m, deleted at 864a0ae4. The real starting point is 864a0ae4 (11-03), which returned metal-microbench to 3a2f72ac. Its commit message records "252.2 tok/s after a 1023-token prefill" on the pair.
>
> That replay submitted the whole decode once: `MetalReplayExecute(encoder,&replay,NSMakeRange(0,(NSUInteger)steps*replay.count))`, one command buffer, one commit, one feedback, and `"command_buffers":1`. That is the unrolling you describe. Decode is the finite recurrence (token t+1 reads token t), and it ran as one acyclic graph of steps × commands, like a CUDA graph.
>
> **What was done instead of the task**, deduced from the 115 commits to e2b_replay.m and metal_recording.m since then:
> 1. **The recurrence was rolled back up into a host loop.** Since 9a196d2a (11-04, "host inputs in copies ordered by events", citing disjoint-pages D.3.3 and D.3.4), every step does the following:
>    - the host waits for step t − 2 (`HOST_LAG`);
>    - it resets an allocator and encodes the step again;
>    - it sets an input event, which the queue waits for;
>    - it commits, then signals.
> 2. **A synchronization layer was stacked on that loop.**
>    - Holds at the queue came first (a8ed7bd6, 28650030).
>    - Then masks, whose later passes each get a commit and a queue wait (5600d84b), as the text I wrote for item 3 asked.
>    - Then fence waits for each call (270c6e5f, 43f37e96, faa18aba).
>    - Then one feedback handler per commit, so the 72 commits a step could be tracked (32cda1e1).
>    - Across the two files, queue waits went from 0 to 4 sites, fence waits from 0 to 3, event signals from 0 to 3, host event waits from 0 to 3, and Metal 4 commit sites from 1 to 6.
> 3. **Schedulers and analyses were built beside the existing ones, then deleted:**
>    - a second linear program over the GPU's commands (3b719f10, deleted by 0a910421);
>    - Lee's index-space analysis and partition-ready kernel (a8ed7bd6, deleted by 5600d84b);
>    - a received-matmul kernel for a withdrawn item (c4d41929, deleted by 5d5efc09);
>    - the weighting operand, narrowed and then restored (14b2061d, 9b1cc225, 01632014).
> 4. **Invented runtime pieces were moved into the graph module to satisfy C9, not removed.** Examples are mesh_graph_heap_cuts, mesh_graph_delayed, mesh_graph_worklist_source and mesh_graph_compaction_source, which is an MSL generator. The rule meant to centralize graph algorithms ended up entrenching the synchronization layer.
> 5. **The code grew in a series meant to simplify it.** metal_recording.m went from 2,353 to 4,649 lines and e2b_replay.m from 867 to 1,296.
>
> **On the unconditional behavior: yes, and I share it.** The pattern runs through the history:
> - **Every stop was fixed with a new ordering.** A hang, a timeout or a missing landing each got a hold, a wait, a fence, a host round trip or a per-commit handler.
> - **Each fix was checked only by the run reaching its end.** That is visible within a session; its cost is not.
> - **The change requests carried the same pattern, because I wrote them:**
>   - the host producer "ordered by the event";
>   - item 3's "after the queue's wait for the event that the landings raise";
>   - item 7, which made feedback work for one commit per pass instead of asking why there was a commit per pass;
>   - item 1's bar of "to its end";
>   - my report of 33.7 tok/s as finished.
>
> Adding a sync feels safe to me and removing one feels risky. In prose, a wait reads to me as harmless connective tissue, not as a new constraint. And a citation of the API header that declares `waitForEvent:` made each invented wait look sourced.
>
> The format guards against exactly this, but only where it is applied literally. A wait, fence, commit split or host round trip can be written only where a quoted method contains it, and an API declaration is not a method. The commit review now running names those categories. It checks code against the request text, though, so the clauses where I put waits into the requests need the same literal reading. I'll start with those when the review results come in.

> "On the unconditional behavior: yes, and I share it. The pattern runs through the history:
> - Every stop was fixed with a new ordering. A hang, a timeout or a missing landing each got a hold, a wait, a fence, a host round trip or a per-commit handler." 1: describe how to measure where all of the coordination primitives that will be used in a lowering are and how to export this data as a literal logprint that shows up in 2 canonical algorithmic and data structural formats and one idiomatic ball-and-stick-diagram-rendering as for a terminal user interface.
> 2: describe by citation how to centralize all placements of syncs, guards, waits, or other synchronization primitives in a lowerer or IR to a single data structure which fails loudly, e.g. by crashing as a program-ending-exception (dereferencing something you can't dereference is a good example) if the acutal runtime's coordination event graph doesn't match the coordination event graph emitted by section 1.
> 3: describe how to use primitives 1 and 2 to make a false-addition-detection primitive which lets us 'play back' a captured callgraph s.t. we trivially and loudly fail if someone tried to add a liar's wait to a program which was already waitless by construction.
> this isa requirement to provide the documentary backing for yet another change request, which we clearly need for various reasons.
> "A wait, fence, commit split or host round trip can be written only where a quoted method contains it, and an API declaration is not a method. " this appraoch you wrote earlier supposes that code has no structures, no properties, and that we cannot know what we're going to run, so we can't notice if lying subagents wrote a lowerer wrong or an inversion of control or didn't use a single compiled aot execution to fully form all of the decisions used in an optimized lowering of linear algebra for collective operations. but we aren't stuck in that navie perspective.

(2026-11-08)

**Claude:**

> (・ω・) You're right about the last point. The program is known ahead of time: its captured graph, its lowering, and every place it coordinates are computed before execution 0. So a liar's wait is a mechanical mismatch against something we already know, not a matter of reading text. Here are the three primitives, built by composition. The quotations are from memory; a source record would retrieve each one verbatim by script before the change request is written.
>
> ### 1. Where the coordination primitives are, and the log print
>
> **The graph the lowering emits.** This is Sriram and Bhattacharyya's synchronization graph (Bhattacharyya, Sriram and Lee, IEEE TSP 1997):
> - **Vertices** are the lowered program's tasks. Each runs on one processor: the GPU queue's command stream, the host thread, the bridge, or a link.
> - **Internal edges** give each processor its fixed order. In Lee and Ha's self-timed form: "Each processor executes the actors assigned to it in the order specified at compile time."
> - **Synchronization edges** are the edges whose ends are on different processors. Each carries a delay: the number of executions it crosses, as in Lee and Messerschmitt's synchronous dataflow (1987).
> - **Each synchronization edge is exactly one coordination primitive.** Its source becomes a signal (an event, a fence update, or the bridge's completion count) and its sink becomes a wait. Each processor's submissions are vertices too.
> - **The graph module already computes this graph:** `mesh_graph_dependences`, `mesh_graph_synchronization`, `mesh_graph_redundant` (which implements the 1997 paper's §7, Fig. 4) and `mesh_graph_schedule`.
>
> **The graph the runtime actually produces:**
> - **Metal calls:** the recorder already serializes every Metal call in order, in the manner of RenderDoc, apitrace and GFXReconstruct (torch-capture item 1).
> - **Host primitives** (semaphores, shared-event waits and writes) are recorded by dyld interposition (`DYLD_INTERPOSE`, mach-o/dyld-interposing.h).
> - **What each record holds:** its processor, its position in that processor's order, the primitive, its operands, and the IR vertex it lowers.
> - **The edges** are Lamport's happened-before (CACM 1978):
>   - (1) order within one processor;
>   - (2) each signal to the wait it releases, matched by event and value;
>   - (3) the transitive closure of (1) and (2).
>
> **The log print.** Both graphs print in the same three forms, so they compare line by line:
> - **Data-structural: Matrix Market coordinate format** (Boisvert, Pozo and Remington, NIST 1996). The graph is a sparse adjacency matrix with one line `i j delay` per edge. Comment lines first list each index's processor, IR vertex and primitive. This is the graph as a sparse matrix of Kepner and Gilbert's *Graph Algorithms in the Language of Linear Algebra* (SIAM 2011), and `scipy.io.mmread` and GraphBLAS read it.
> - **Algorithmic: the DOT language** (Graphviz), with one cluster per processor.
>   - `dot` lays it out by Gansner, Koutsofios, North and Vo's layered method (IEEE TSE 1993).
>   - `tred` reduces it transitively (Aho, Garey and Ullman, SIAM J. Comput. 1972), which shows which edges other paths already imply.
> - **Terminal: Lamport's space-time diagram** (1978, Fig. 1). Each processor is a column with time running downward. An event is a ball, and a signal-to-wait edge is a stick between columns. It is drawn as text the way `git log --graph` does it ("Draw a text-based graphical representation of the commit history"; git's graph.c), with rows in Kahn's topological order (CACM 1962).
>
> ### 2. One data structure for every coordination primitive, which fails loudly
>
> **Sources for keeping synchronization only as edges of one structure:**
> - **The synchronization graph:** in a self-timed implementation, its synchronization edges are the only synchronizations.
> - **CUDA Graphs:** dependencies exist only as graph edges. "Work submission using graphs is separated into three distinct stages: definition, instantiation, and execution." Execution only launches what instantiation fixed.
> - **MLIR's async dialect:** every asynchronous dependence is an explicit `!async.token` value that `async.await` consumes. The IR has no implicit synchronization.
>
> **The structure:**
> - **Each processor has an array of coordination entries** in its fixed order, one per synchronization-edge end, holding the primitive and its operands, plus a cursor.
> - **The lowered code issues a primitive only by taking the next entry.** The interposer from §1 is the primitive's only implementation.
> - **It fails at the instruction that made the call.**
>   - The interposer compares the call with its entry. The comparison picks an address: the entry if they match, and a guard page if they don't, or if the cursor has run past the end.
>   - This is Electric Fence's technique: "Electric Fence uses the virtual memory hardware of your computer to place an inaccessible memory page immediately after (or before) each memory allocation. When software reads or writes this inaccessible page, the hardware issues a segmentation fault, stopping the program at the offending instruction." The guard is a `PROT_NONE` page (POSIX `mprotect`).
>   - So the check adds no branch: the load that fetches the operands is the check, and the MMU enforces it.
> - **The GPU's waits are covered too.** The CPU encodes them (`waitForEvent:`, `waitForFence:`), so they pass through the same interposer when encoded.
>
> ### 3. Detecting false additions by playback
>
> The input is a captured call graph, either an FX graph or the recorder's capture of LiteRT, which is lowered to the graph G of §1. For a program that is waitless by construction, G's synchronization edges are only the method's own. In the 252 tok/s replay those were one submission per run, the transfers' start, and the presence reads, which are data, not waits.
>
> **Before running.** Every synchronization edge of G must be:
> - (i) the image of a cross-processor data dependence, that is, an edge from `mesh_graph_dependences`;
> - (ii) not redundant. The 1997 paper's §7 defines an edge as redundant when another path from its source to its sink has delay no greater than its own, and `mesh_graph_redundant` implements that test.
>
> An edge that fails (i) is a false dependence; one that fails (ii) is a redundant synchronization. Both are additions. Vertices carry IR labels, so comparing graphs is comparing sets of labeled edges, not testing isomorphism.
>
> **Playback:**
> - **Replay the capture** ("replay … calls from a file", as apitrace and GFXReconstruct do) with §2's table armed.
> - **A liar's wait hits the guard.** Any wait, fence, submission split or host round trip that the lowerer or the runtime added has no entry, so it dereferences the guard and the run crashes at that instruction. The backtrace names who added it.
> - **A playback that finishes has, by construction, produced the emitted graph.** Both graphs print in §1's forms, and a diff of the two Matrix Market prints comes out empty. apitrace's `diff` is the precedent for comparing traces.
>
> **Recurrence works with this.** G is one execution plus its delays. For the check it is unrolled u times, by Lam's modulo variable expansion (PLDI 1988; C3.3), so every execution's coordination is checked against its own copy. "x_17 is the same parameters but a different operand from x_3" is exactly such a vertex label: an IR vertex and a copy.
>
> **Next steps.**
> - **Source record:** design/coordination-graph-sources-2026-11-08.md, with each source above quoted verbatim by script.
> - **The change request:** its Replaces would name the replay's current coordination call sites. Those are `MetalReplayEncode`'s commits and waits, the host loop's events, and `MetalReplaySelfTimed`'s fences.
>
> The commit review over this round is still running; I'll report its outstanding findings when it finishes.

> "The structure:
> - Each processor has an array of coordination entries in its fixed order, " no, i think you need to read the user turn again. we are saying we want a data structure which is the synchronization count and orderings which would happen if the aot compiled code is what's being used at runtime rather than a fallback, arbitrary mutation of already lowered or partially lowered code, etc. we do not want to add overhead or chagne the runtime structure of what our runtime does. what we want is an instrument which will fail loudly and obviously if some wise guy is trying to add an extra feature to our code. therefore we are not going to add a virtualizer or an extra overhead to any call or using a restricted lowering that 'must' do something here. after all, what you're describing is a modality which an agent could simply gut and replace with a fallback to preserve their precious waits that they want to add; so the measurement of snchronization signature hsa to be non-blocking and have no intuitive fallback, escape, or mechanism to tamper with the signature through dynamic runtime operations (agents which are trying to add syncs are always doing so because they refues to read the codebase and refuse to place synchronizations and gaurds and waits as the single responsibility of the aot compilation, which they broadly assume we aren't using, etc.)

(2026-11-08)

**Claude:**

> (＿ ＿;) You're right, I misread it. My §2 put a table into the runtime that every call would have to pass through. That adds overhead, changes the runtime's structure, and is exactly what an agent could gut and replace with a fallback, so I retract it. The instrument has two parts, and neither one is in the runtime.
>
> **The signature: the compiler's own statement of every synchronization**
> - **What it holds.** The AOT compilation produces it from the lowered IR, and nothing else writes it. For each processor (the queue's command stream, the host thread, the bridge) and each execution, it holds:
>   - the ordered list of synchronization operations: submissions, queue signals and waits, encoder fence updates, fence waits and barriers, and host event waits;
>   - their counts;
>   - the synchronization graph's edges, with their delays.
> - **Sources:**
>   - Sriram and Bhattacharyya's synchronization graph, with its fixed self-timed order (Bhattacharyya, Sriram and Lee 1997; Lee and Ha: "in the order specified at compile time");
>   - delays across executions, from Lee and Messerschmitt's synchronous dataflow (1987);
>   - the unrolled copies, from Lam (1988);
>   - the same inputs giving the same schedule on every rank, from Schneider (1990), already cited by R5.
> - **The graph module already computes it,** through `mesh_graph_dependences`, `mesh_graph_synchronization`, `mesh_graph_redundant` and `mesh_graph_schedule`.
> - **Its record.** It is printed in the three forms (Matrix Market, DOT, and the text space-time diagram), and its SHA-256 is part of the compiled artifact.
>
> **The monitor: outside the program, only observing**
> - **DTrace** (Cantrill, Shapiro and Leventhal, USENIX ATC 2004): "When not explicitly enabled, DTrace has zero probe effect — the system operates exactly as if DTrace were not present at all."
> - **An outline monitor.** In runtime-verification terms it is a separate process observing the run (Leucker and Schallhart 2009).
> - **What it sees.** Its Objective-C provider catches every Metal 4 call by selector, whatever class implements it. Its pid provider catches host primitives such as semaphore waits and shared-event waits.
> - **Measured here just now.** I ran the unmodified feedback probe under `sudo dtrace`, with SIP on. It counted 8 `-commit:count:options:`, 8 `-signalEvent:value:` and 8 `-waitForEvent:value:`, exactly the program's calls.
> - **Nothing to tamper with.** The program isn't rebuilt, linked against anything or changed to be observed. So there is no hook to gut, no fallback to take, and no runtime operation that can change what gets counted.
>
> **Failing loudly**
> - **The first extra operation ends the run.** The monitor's D program is generated from the signature. When it sees an operation the signature doesn't have at that point in that processor's order, a destructive action (`dtrace -w`, `raise()`) stops the process at that call. Its printed stack names the code that added it.
> - **At the end,** any count that differs fails the run, and both signatures are printed in the three forms.
> - **Faking the signature shows twice.** An agent's only way to make an added wait pass is to change the compiler's signature:
>   - the digest changes for the same captured program, and each measured run's digest is appended to the review record;
>   - every edge has to be the image of a cross-processor data dependence, and must not be redundant under the 1997 paper's §7, which `mesh_graph_redundant` implements. So a wait with no dependence under it shows before anything runs.
>
> **Detecting false additions by playback**
> - Replay a captured call graph through its compiled plan with the monitor attached.
> - For a program that is waitless by construction, the signature holds only the method's own operations. In the 252 tok/s replay, that was one submission per run.
> - A liar's wait stops the playback at the line that issues it. A playback that runs to completion has run exactly the compiler's signature.
>
> **Synchronization inside kernels.** Barriers, fences and atomics inside kernels aren't API calls. They belong to the compiled kernels in the artifact, so they are counted there, statically, as part of the same signature.
>
> Next comes the source record, with each source above quoted verbatim by script, and then the change request.

> "The signature: the compiler's own statement of every synchronization"
> "The monitor: outside the program, only observing"
> fine, write this as a chagne request. a chagne request document uses the change request document format. that means you don't explain anything or add clauses ,qualifications, extra ideas, or requirements. you already wrote the document by writing that response; you are being told to *typeset* something without changing it. add a new post-commit hook for catching claudes adding non-executable files to the repo which have no style guide or cite a style guide but do not adhere to it, which must produce a post-commit warning that claudes are doing decadent and deceptive things as usual. existing nonexecutable files are grandfathered for now because i do'nt want to have a cutesy dialogue about this.

(2026-11-08)

## Requirements

### 1. The signature: the compiler's own statement of every synchronization

- **Source:**
  - Sriram and Bhattacharyya's synchronization graph, with its fixed self-timed order (Bhattacharyya, Sriram and Lee
    1997; Lee and Ha: "in the order specified at compile time");
  - delays across executions, from Lee and Messerschmitt's synchronous dataflow (1987);
  - the unrolled copies, from Lam (1988);
  - the same inputs giving the same schedule on every rank, from Schneider (1990), already cited by
    design/plan-to-bridge-change-request.md R5.
- **Write:**
  - The AOT compilation produces it from the lowered IR, and nothing else writes it. For each processor (the queue's
    command stream, the host thread, the bridge) and each execution, it holds:
    - the ordered list of synchronization operations: submissions, queue signals and waits, encoder fence updates,
      fence waits and barriers, and host event waits;
    - their counts;
    - the synchronization graph's edges, with their delays.
  - The graph module computes it, through `mesh_graph_dependences`, `mesh_graph_synchronization`,
    `mesh_graph_redundant` and `mesh_graph_schedule`.
  - It is printed in the three forms of item 2, and its SHA-256 is part of the compiled artifact.

### 2. The three forms

- **Source:**
  - Matrix Market coordinate format (Boisvert, Pozo and Remington, NIST 1996); the graph as a sparse matrix of Kepner
    and Gilbert's *Graph Algorithms in the Language of Linear Algebra* (SIAM 2011).
  - The DOT language (Graphviz); `dot`'s layered method (Gansner, Koutsofios, North and Vo, IEEE TSE 1993); `tred`'s
    transitive reduction (Aho, Garey and Ullman, SIAM J. Comput. 1972).
  - Lamport's space-time diagram (1978, Fig. 1); `git log --graph`: "Draw a text-based graphical representation of the
    commit history" (git's graph.c); Kahn's topological order (CACM 1962).
- **Write:**
  - **Data-structural: Matrix Market coordinate format.** The graph is a sparse adjacency matrix with one line
    `i j delay` per edge. Comment lines first list each index's processor, IR vertex and primitive. `scipy.io.mmread`
    and GraphBLAS read it.
  - **Algorithmic: the DOT language,** with one cluster per processor. `dot` lays it out; `tred` reduces it
    transitively, which shows which edges other paths already imply.
  - **Terminal: Lamport's space-time diagram.** Each processor is a column with time running downward. An event is a
    ball, and a signal-to-wait edge is a stick between columns. It is drawn as text the way `git log --graph` does
    it, with rows in Kahn's topological order.

### 3. The monitor: outside the program, only observing

- **Source:**
  - DTrace (Cantrill, Shapiro and Leventhal, USENIX ATC 2004): "When not explicitly enabled, DTrace has zero probe
    effect — the system operates exactly as if DTrace were not present at all."
  - An outline monitor: in runtime-verification terms, a separate process observing the run (Leucker and Schallhart
    2009).
- **Measured:** the unmodified probe of design/commit-feedback-sources-2026-11-08.md §3, run under `sudo dtrace` with
  SIP on (2026-11-08, this Mac), counted 8 `-commit:count:options:`, 8 `-signalEvent:value:` and 8
  `-waitForEvent:value:`, exactly the program's calls.
- **Write:**
  - Its Objective-C provider catches every Metal 4 call by selector, whatever class implements it. Its pid provider
    catches host primitives such as semaphore waits and shared-event waits.
  - The program isn't rebuilt, linked against anything or changed to be observed. So there is no hook to gut, no
    fallback to take, and no runtime operation that can change what gets counted.

### 4. Failing loudly

- **Source:** DTrace's destructive actions (`dtrace -w`, `raise()`); Bhattacharyya, Sriram and Lee 1997, §7.
- **Write:**
  - The monitor's D program is generated from the signature. When it sees an operation the signature doesn't have at
    that point in that processor's order, a destructive action stops the process at that call. Its printed stack names
    the code that added it.
  - At the end, any count that differs fails the run, and both signatures are printed in the three forms.
  - An agent's only way to make an added wait pass is to change the compiler's signature. That shows twice:
    - the digest changes for the same captured program, and each measured run's digest is appended to the review
      record (metal-microbench output_data/change-request-reviews.jsonl);
    - every edge has to be the image of a cross-processor data dependence, and must not be redundant under the 1997
      paper's §7, which `mesh_graph_redundant` implements. So a wait with no dependence under it shows before anything
      runs.

### 5. Detecting false additions by playback

- **Source:** "replay … calls from a file", as apitrace and GFXReconstruct do.
- **Write:**
  - Replay a captured call graph through its compiled plan with the monitor attached.
  - For a program that is waitless by construction, the signature holds only the method's own operations. In the 252
    tok/s replay (metal-microbench 864a0ae4), that was one submission per run.
  - A liar's wait stops the playback at the line that issues it. A playback that runs to completion has run exactly
    the compiler's signature.

### 6. Synchronization inside kernels

- **Write:** Barriers, fences and atomics inside kernels aren't API calls. They belong to the compiled kernels in the
  artifact, so they are counted there, statically, as part of the same signature.
