# Change request: the synchronization signature and its monitor

Code is cited at mesh e8cd57e0 and metal-microbench 21c44310.

## The turns

> "The sources quoted in the dialogue already rule the overheads out:" so we need to deduce what implementing agents have been doing besides refactoring, simplifying, and centarlizing the source of the rpoject upon its rqeuirements to consume a callgraph, fxgraph parse and lower it through a lowerer, ensuring the lowerer has certain properties, then run the lowered code (with no other entry points to the collective opwerations we measure than this singular simple path); using the optimizing compiler features described in the citations to *choose* lowerings which are compatible iwth all of the goals we have, to *implement* lowerings which are compatibel iwth the properties of the dma driver and its necessary wrapper code, compatible iwth the properties of linear algebra and the dependency structures we know ahead of time about linear algebraic functions which are executed in particular orders in particular callgraphs. what do we feel about recurrence here? well, like cudagraphs adn friends we implicitly took this position that we only parse or handle directed graphs with finite numbers of (directed) cycles, s.t. we can explicitly unroll all cycles to bigger directed acyclic graphs with very convenient and obvious and compiler-IR-legible type annotations like 'x_17 is the same parameters but a different operand from x_3 in the 18-deep weight-tied depthwise-recurrent RESNET_18 we're using'. (RESNET_18 is a purely hypothetical architecture, you're not in a kaiming he jumpscare horror format here).
>
> "None of these contains a commit and a queue wait before every later pass, "
> consider highreflectinguinely that claudes have an unconditional behavior of always trying to add queue waits and allsyncs and serializations to partially optimized (or even fully optimized) async networking code, and that all of the documentary structure of the chagne request format and similar is here to mitigate and channel claudes away from unconditional behavior.

(2026-11-08)

> "On the unconditional behavior: yes, and I share it. The pattern runs through the history:
> - Every stop was fixed with a new ordering. A hang, a timeout or a missing landing each got a hold, a wait, a fence, a host round trip or a per-commit handler." 1: describe how to measure where all of the coordination primitives that will be used in a lowering are and how to export this data as a literal logprint that shows up in 2 canonical algorithmic and data structural formats and one idiomatic ball-and-stick-diagram-rendering as for a terminal user interface.
> 2: describe by citation how to centralize all placements of syncs, guards, waits, or other synchronization primitives in a lowerer or IR to a single data structure which fails loudly, e.g. by crashing as a program-ending-exception (dereferencing something you can't dereference is a good example) if the acutal runtime's coordination event graph doesn't match the coordination event graph emitted by section 1.
> 3: describe how to use primitives 1 and 2 to make a false-addition-detection primitive which lets us 'play back' a captured callgraph s.t. we trivially and loudly fail if someone tried to add a liar's wait to a program which was already waitless by construction.
> this isa requirement to provide the documentary backing for yet another change request, which we clearly need for various reasons.
> "A wait, fence, commit split or host round trip can be written only where a quoted method contains it, and an API declaration is not a method. " this appraoch you wrote earlier supposes that code has no structures, no properties, and that we cannot know what we're going to run, so we can't notice if lying subagents wrote a lowerer wrong or an inversion of control or didn't use a single compiled aot execution to fully form all of the decisions used in an optimized lowering of linear algebra for collective operations. but we aren't stuck in that navie perspective.

(2026-11-08)

> "The structure:
> - Each processor has an array of coordination entries in its fixed order, " no, i think you need to read the user turn again. we are saying we want a data structure which is the synchronization count and orderings which would happen if the aot compiled code is what's being used at runtime rather than a fallback, arbitrary mutation of already lowered or partially lowered code, etc. we do not want to add overhead or chagne the runtime structure of what our runtime does. what we want is an instrument which will fail loudly and obviously if some wise guy is trying to add an extra feature to our code. therefore we are not going to add a virtualizer or an extra overhead to any call or using a restricted lowering that 'must' do something here. after all, what you're describing is a modality which an agent could simply gut and replace with a fallback to preserve their precious waits that they want to add; so the measurement of snchronization signature hsa to be non-blocking and have no intuitive fallback, escape, or mechanism to tamper with the signature through dynamic runtime operations (agents which are trying to add syncs are always doing so because they refues to read the codebase and refuse to place synchronizations and gaurds and waits as the single responsibility of the aot compilation, which they broadly assume we aren't using, etc.)

(2026-11-08)

> "The signature: the compiler's own statement of every synchronization"
> "The monitor: outside the program, only observing"
> fine, write this as a chagne request. a chagne request document uses the change request document format. that means you don't explain anything or add clauses ,qualifications, extra ideas, or requirements. you already wrote the document by writing that response; you are being told to *typeset* something without changing it.

(2026-11-08, in part)

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
