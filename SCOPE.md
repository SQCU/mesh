# What the library has to do

The operator's statements, verbatim, then Claude's answers to the operator's questions, verbatim (2026-11-03).

## Composition, not computation

The operator, 2026-11-03:

> the difference between composition (we can compile `a()\nBLOCK\nb()BLOCK` to `partial:b(partial(a()))` while still
> treating a() and b() totally opaquely) and computation is simple and serious; redefining what a call does requires the
> meshlib to expand to hundreds of thousands of lines of code to 'reproduce' the behavior of thousands of lines of codes
> of backends. refusing to define what calls do other than slicing and viewing and parallelizing them means the number
> of lines of code in meshlib can stay constant as the number of lines of codes of programs supported scales to
> millions or billions.

> broadly, i want the meshlib to not define how to compute any linear algebraic function and exclusively take the
> responsibility of defining how collective operations are load balanced across devices, how transactions are load
> balanced along edges, how memory on different machines is consistently described for aot compilation (paging), and
> leaving definitions of functions that run over pages to exist entirely in the data processing headers, which should
> really only be there as helpers for accessing tensors, zero copy, by view, rather than reimplementing or inlining any
> arithmetic related code.

> hmmm... okay. i think ther's still something confused here about the graph capture. doing graph capture has a specific
> relationship to compiling torhc programs into runtimes which don't GIL lock and induce latency on dma transactions.
> there are some requiements for the library code to wrap, transduce, and perhaps operation-fuse programs. but that
> doesn't mean that inlined mps backends are part of the library either, i don't think...

## The limits

The operator: "what i am describing as a list of limits dramatically reduces the size and complexity of everything here.
please write up what the limits are and whether or not tehy reduce the scope of complexity or testing or validation or
maintenance for the mesh lib and when they would be desirable or undesirable."

**The limits.** The mesh library is responsible for three things and nothing else:

1. **Balancing collective work across devices.** How much of each collective's data and work each device holds, from measured rates.
2. **Balancing transactions along edges.** Which pages cross which cables and relays, and in what order, so every link is busy and none is the bottleneck.
3. **Describing memory consistently across machines for AOT compilation.** Page names, each node's page table and page lists, so both ends of every transfer agree before anything runs.

Excluded from the mesh library:
- any definition of a linear-algebra function: kernels, matmul, attention, pointwise ops;
- any arithmetic, including the combine step of a reduction;
- running or recording the program.

The data-processing header only gives zero-copy views over pages, contiguous or paged, for the program's own kernels.

**Scope.** Almost all of it goes. The library no longer needs:
- operator coverage or op lowering;
- fusion or dtype cases;
- attention variants;
- following changes in framework operators.

What remains is a planner whose input is the fabric, the rates and the declared calls, and whose output is shares, routes and page lists, plus the bridge mechanics.

**Complexity.** It collapses to a few well-studied problems, each with literature to cite:
- partitioning for load balance;
- multi-commodity routing and tree packing (the linear program and Garg-Könemann);
- placement by lifetime (offline memory planning);
- one order (list scheduling).

No numerical state and no per-op behaviour is left.

**Testing and validation.** They become structural, and you can check them without numbers:
- both ends' page lists agree;
- every page lands where it was named;
- no page is overwritten while something still reads it (happens-before);
- routes come close to the linear program's bound;
- shares come close to the balanced makespan.

Numerical validation stays with the frameworks, whose kernels are already validated. End-to-end you still have to show that a real program over views runs at the hand-written latency, but nothing per op or per dtype.

**Maintenance.** The library changes only when the fabric or the driver does. Kernel performance belongs to the frameworks and to Apple.

**When the limits are desirable:**
- The program's kernels already exist and are fast (torch MPS, MLX, LiteRT, hand-written kernels) and can read views: strides, pointers, block tables. Paged attention already works that way.
- The cabling is arbitrary or heterogeneous, so the hard problem really is balancing and routing.
- Shapes are static, so paging can be decided ahead of time.

**When they cost something:**
- **Reductions:** the combine is arithmetic, so the program has to supply it. A library that fuses reduction into transport, as NCCL does inside its kernels, can save a dispatch. The better form here is for the consuming kernel to sum the partials as it loads them through paged views, so the cost becomes a design obligation on the program rather than a loss.
- **Frameworks whose kernels need contiguous tensors:** they copy, outside the library. Placement can reduce that by giving contiguous-typed operands one contiguous run.
- **Overlap across operator boundaries** (consuming pages as they land, before the whole collective finishes) needs the program's kernels to use the per-page guards. The library supplies the guards but can't force their use.
- **Data-dependent routing** (MoE expert choice, dynamic shapes) has to come in as routing operands over declared ranges. The library can take those but can't decide from values.

## Whether the documentation states these limits

The operator: "are these requirements deducible from all of the documentation? don't read any of it, vibe check only.
are these requirements successfully derived from current documentation, including self-documentation claudes pass
along?"

No, not as stated, though the pieces are there.

**What points toward your limits:**
- `data-model.md` §5-6 (the routing function returning shares and a plan of page writes, page lists, a page table per node). That is limits 1-3 in all but name.
- The rubric's interface table, especially "between mesh and data: nothing".
- The deliverables' invariants that keep numerics and application code out of mesh.

A careful reader could get to "mesh decides balance, routes and paging" from those.

**What points the other way, louder:**
- **Kernels classed as collective code.** The rubric files torch-mesh, with its kernels and compiler, as collective code. So defining linear-algebra kernels reads as the library's job.
- **The program surface:** the partials-merge operator, combine kernels, fused forms, and the compiler lowering torch ops into Metal kernels. All of it says the library computes.
- **Claude-written material:** facts.md, the memory notes and the dated design records are mostly kernel engineering. Product forms, fusion, matmul2d tiling, fold dispatches and pointwise codegen are all recorded as the library's open work. The "integration closing set" note (all FX/AOT torch, LiteRT, MPS programs) reads as "the library's compiler must lower every program", the opposite of "define no linear algebra".

So the documentation is consistent with your limits but doesn't state them as limits, and the Claude-written layer has drifted far from them. Every session, mine included today, followed that drift: tuning products, fusing cells, writing a fold kernel.

## Routed operations (MoE) are static in shape

The operator: "describe moe experts again as anm indexed sparse kernel instead of a data dependent kernel. are moe
operations or routed operations dynamic in shape? yes or no."

No. MoE and other routed operations are not dynamic in shape.

An MoE layer is an indexed sparse kernel with static shapes:
- **The router** outputs a fixed-shape index tensor, top-k expert ids per token, `[tokens, k]`, plus gates `[tokens, k]`.
- **Dispatch** is a gather by those indices into each expert's fixed-capacity buffer, `[experts, capacity, hidden]`.
- **The expert computation** is a grouped or indexed GEMM over that fixed buffer.
- **Combine** is a scatter-add back to `[tokens, hidden]`, weighted by the gates.

Every shape is known ahead of time. Only the index values vary, and they're operands, the same as the values in any matmul.

So I was wrong to call it data-dependent routing that the library can't decide. Across machines, each expert's capacity pages are fixed, so the page lists and routes are fixed and plannable ahead of time like any other collective. The index tensor is just data. The kernels read rows of those pages through indexed views, the way an embedding lookup or paged attention's block table does. Nothing in the plan depends on what the router outputs.

## Any torch program, and a library that only pages and transacts

The operator: "what does the fx aot torch work semantically mean and what is the relationship between 'running any
arbitrary torch program' and 'the meshlib nevertheless only defines how to page memory and transact memory across
computers'? is there a really obvious synthesis without contradiction here? three sentences only."

FX/AOT capture means torch itself traces the program into a static graph of its own operations, with every tensor's shape and lifetime known. Torch's own kernels compute every value in that graph. So "running any arbitrary torch program" is torch's capability, and the meshlib's part is only to read that graph's tensors and collective calls, page every tensor into consistently described region memory, balance the shards across devices, and plan the transfers along the edges, handing torch zero-copy views over those pages. There's no contradiction: the meshlib "lowers" memory and communication for any program and never computation.

## Lowering memory and communication is easy; recording and replay is aligned but was misused

The operator: "is it easy or hard to lower memory and communciation for arbitrary torch programs which are known aot and
do not use data dependent tensor creation or what would be graph breaks in cudagraphcapture if we were using
cudagraphcapture? is the graph recording and replay functionality done earlier in the repo's history opposed to the
repo's goals or mostly aligned but misused by later developers?"

**Easy or hard?** Easy, as these problems go.

With an AOT graph that has no data-dependent tensor creation and nothing that would be a CUDA-graph break, every tensor's size, lifetime and producer and consumer are fixed before the run:
- **Memory** is offline static allocation over known lifetimes. TFLite's arena planner, XLA and CUDA-graph capture solve it routinely (Pisarchyk and Lee's greedy by size).
- **Communication** is the graph's collective calls with known sizes and endpoints, so shares, routes and order can all be computed ahead of time.
- **Handing torch views** over the planned pages is zero-copy storage, which torch already supports.

Nothing has to be decided at run time. The only hard part is quality: how close the shares and routes come to the balance and saturation bounds. Whether it can be done at all isn't in question.

**Recording and replay: opposed to the goals, or misused?** Mostly aligned in intent and misused in scope. This is a vibe judgment; I haven't re-read the history.

Recording once and replaying with new inputs is what the documents ask for. The data model says "Once the plan is set, new input values can follow the same writes". The deliverables cite persistent collectives and CUDA graphs, and torch-mesh's README has kernels recorded before execution.

The misuse came later:
- **Kernels inside the recorder.** The recorder became where computation was defined: kernel generation, product forms, fusion.
- **Scheduling inside the recorder.** It became a scheduler in its own right, with ordering, waits, cuts, rings and slots, at 2,257 lines over 113 functions.
- **Eager and collective-specific paths** grew around it.

Recording and replay is the right mechanism. It was asked to do the meshlib's planning and torch's computing as well, and that's what has to go.

## Capture, transduction and fusion are the library's; computation is not

The operator's statement on graph capture is quoted above.

You're right, my "views only" was too thin. Capture is in the library, for one reason: it's what takes Python and the GIL off the DMA path.

Torch run eagerly dispatches op by op from the interpreter, so every collective's publication, and every consumer waiting on a landing, sits behind host dispatch latency. Capture plus AOT compilation turns the program into a recorded runtime that replays with no interpreter and no lock on the critical path. Only then do transfers and compute move at hardware speed.

So the library's job includes three things the three limits didn't name:
- **Wrap:** be torch's backend, so programs come in through `torch.distributed` and `torch.export`.
- **Transduce:** turn the captured graph into a recorded, replayable Metal stream. Torch's own kernels go in, in graph order, over paged buffers, with the collective's publications and landed guards interleaved where the graph puts its collectives.
- **Fuse, where it removes dispatch boundaries:** fewer command buffers, barriers only at real dependences, consecutive word stores batched.

What it still doesn't do is define computation. The kernels come from torch: its MPS backend, or fused kernels from torch's own compiler (Inductor's Metal path), used as delivered. If fusing kernels needs new code, that's torch's compiler writing it, not the library inlining an MPS backend of its own.

That also corrects `core/`. `dataprocessing.h`'s recorder is the right kind of thing, because transducing a graph into a recorded stream belongs in the library. What doesn't belong there is its own kernels: the fold, the fill, and anything else that computes.
