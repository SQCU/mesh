# Change request: pages assigned by liveness, landings read by mask from a contiguous list

The operator's and Claude's turns of 2026-11-03 on the page readiness word and the plan's ring, verbatim; the
literature; the state of the code they change (mesh 7e9936d5, metal-microbench f154814d); the data structures and
functions in scope; and the canonical path.

## The turns

**The operator:**

> what's a page readiness word and what overehad does it add in dynamic control flow or data flow during a runtime /
> what overhead does it add in transport related waits, guards, etc?

**Claude:**

> **What a page readiness word is.** It's an 8-byte atomic at the start of a 256-byte cell. Each word gets one cell per
> execution slot of the plan's ring. There are three kinds:
> - **Supplied word:** the producer stores it after writing its pages. One store can release a whole range of pages.
> - **Landed word:** the receiving bridge stores it once that one page's bytes are in place. Each landed word stands for
>   exactly one page.
> - **Relay forward count:** at offset 8 of a landed word's cell, a relay counts the forwards it has completed.
>
> **What it costs at run time:**
> - **Memory:** a 256-byte cell per page per ring slot, about 1.6% of a 16 KiB page.
> - **On the bridge:**
>   - one store per landed page;
>   - one load per send, to check it's released;
>   - at a relay, one load of the forward count before a receive is reposted.
>
>   All of that is non-blocking, and no new guards were added. All-ones still cancels.
> - **On the GPU, where the real cost is:** a reader waits on each page's own landed word. Every reading thread does a
>   system-scope fenced load the first time it touches a page, so a thread that walks k pages pays k fenced loads
>   before computing, even when everything has already landed. In 10-01, one word stood for a whole chunk on one queue
>   pair, so readers waited once per chunk. That multiplication is the price of spreading a transfer's pages across
>   every cable.
> - **The fix I'd propose:** a range's landed word counts pages. Each landing adds one, atomically, and a reader waits
>   once for the full count. Pages can still land in any order, and the per-page waits go away. The trade-off is that
>   the word gets several writers, one per port child, instead of one.

**The operator:**

> "plan's ring."
> what's a ring and what do you think a ring does?
> "That multiplication is the price of spreading a transfer's pages across every cable."
> i think you are descirbing an extremely nonstandard data structure and an extremely nonstandard algorithm for
> checking something which does not in any sense have to be embedded into a struct in this manner. rather than trying to
> rush into any sort of deletion or change, you need to cite from the literature what prior art this very specific
> implemetnatino descends from most closely and what it is optimizing for algorithmically. it can't not be optimizing
> for *something*, but i suspect oyu will find what is being optimized is not named or forthrightly described. i do not
> want fixes before citation.

**Claude:**

> **2. The "ring."**
> - **What it is:** a plan that runs many executions (the decode's steps) doesn't get fresh pages for each one. It
>   reuses `depth` copies of each page, with execution t using slot `t mod depth`.
> - **The two waits it imposes:**
>   - execution t's receive can't be posted until executions up to `t − (depth − 1)` have landed;
>   - from `t = depth` on, a receive also waits until the relay has forwarded slot `t − depth`'s page.
> - **What it does:** it keeps memory at `depth` slots instead of growing with the number of executions. In exchange,
>   each execution waits on an earlier one before its pages can be reused.
> - **Where it comes from:** the 10-01 code's own `MESH_RING_DEPTH` (the decode in 16-step windows), carried forward.
> - **How it relates to what you said:** it's the pattern you described as scheduling buffers "to land in the exact
>   same spot on every repetition instead of swizzling or striding across the full addressable pageset".
>
> **3. Where the per-page readiness word comes from, and what it optimizes.**
> - **Closest prior art: presence-per-element memory.**
>   - Arvind, Nikhil and Pingali's I-structures (ACM TOPLAS 11(4), 1989): write-once elements, each with a presence
>     bit, where a read of an unwritten element is deferred until it's written.
>   - The hardware forms before that: the full/empty bits of Smith's HEP (1981) and later the Tera MTA.
>   - In GPU collectives: NCCL's LL and LL128 protocols, which put a flag beside every 8 bytes or every 128-byte line
>     so a receiver can poll for arrival without a separate fence.
> - **The road not taken: counted completion.**
>   - Portals 4's counting events (wait until a counter reaches a threshold).
>   - OpenSHMEM 1.5's put-with-signal with `SHMEM_SIGNAL_ADD`, and NVSHMEM's equivalent: one word, many writers adding
>     to it, one wait for the total.
>   - Belli and Hoefler's notified access for MPI one-sided communication (IPDPS 2015).
> - **What it optimizes, though the code never says so:**
>   - **Consumption granularity:** a reader can use each page the moment it lands, at the cost of one wait per page.
>   - **Exactly one writer per word:** no atomic read-modify-write between port children on different cables.
> - **What the code says instead:** it gives only the correctness reason, that pages "land in any order, unlike main's
>   chunks on one queue pair, whose last word stood for all".
> - **The 256-byte cell** is padding so no two words share a cache line. It's the standard false-sharing avoidance,
>   also unnamed.
> - **The ring itself** descends from the bounded buffer (Dijkstra, "Cooperating Sequential Processes", 1965). In
>   collectives that's NCCL's per-connection FIFO of `NCCL_STEPS` slots with head and tail counters. It optimizes for
>   bounded memory and for receives posted one window ahead.

**The operator:**

> "- What it does: it keeps memory at depth slots instead of growing with the number of executions. "
> yeah nobody asked for this nad if you carefully note the definitions of topics nad problems. really carefully note
> them. note them really carefully. you'll find that 'we need to free all memory used as fast as possible even if it
> means we hide 20 inlined allocators using a modulo algebra adn NOTHING ELSE to free memory and add a wait or
> synchronization before all steps' was never transmitted from the user to an agent, any agent, ever... in fact hte
> opposite message was written many times, in many ways, in many forms, in ever more qualitative detail in case anyone
> tried to get clever and pretend to miss the point. describe what the algorithm is minimizing and what it is
> maximizing in greater detail. then describe what the data model says the hardware literally is like.
> " at the cost of one wait per page."
> you will notice something is really stupid about this.

**Claude:**

> **What the ring and per-page words minimize:**
> - the memory a repeating plan holds: `depth` copies of each page, however many executions run;
> - the receives outstanding at once: at most `depth − 1` executions ahead;
> - the writers per word: exactly one;
> - the delay before a reader can start on any single page.
>
> **What they maximize:**
> - reuse of each page, every `depth` executions;
> - consumption granularity, page by page.
>
> **What that costs:**
> - a synchronization before every execution once the ring wraps, because execution t waits on t − (depth − 1)'s
>   landings and t − depth's forwards;
> - a fenced check for every page by every thread that reads it.
>
> It optimizes nothing about link use, call latency or the absence of waits.
>
> **What you asked for instead,** many times:
> - "bin-pack over pages to ensure one-sided checkless waitless syncless writes, by ensuring nonoverlap of adjacent
>   iterations of a repeating function".
> - Not landing "in the exact same spot on every repetition instead of swizzling or striding across the full
>   addressable pageset, freeing whatever was used in previous iterations whenever that's asynchronous and nonblocking
>   of any other algorithms".
> - Slots as "craven attempts to reprotocolize a ring buffer or some vibes-based theory of finite and unmanaged or
>   privately managed capacity (rather than the page table abstraction of managed and checked capacity!)".
>
> So the ring minimizes the one thing never requested, memory, and pays for it with exactly what was forbidden: waits
> and synchronization between steps.
>
> **What the data model says the hardware literally is:**
> - **One memory.** "No HBM, no BAR and no PCIe peer-to-peer path"; a node's "compute units and links share memory".
> - **Registered once, and nearly all of it.** "A Studio wires 82 of its 96 GiB this way, as 82 extents of 1 GiB …
>   Every page a program uses lies in that region, so nothing is pinned at a call."
> - **The driver:**
>   - SEND and RECV only, on unreliable-connected queue pairs, with no remote write, read or atomics;
>   - a SEND fills the oldest receive posted on its queue pair, and both must span the same number of 4 KB credit
>     frames;
>   - the hardware holds a SEND in credit until its receive is posted, and sends no acknowledgement;
>   - one range per request, 100 registrations, one line of receives per queue pair, no shared receive queue.
> - **How it's meant to be used:**
>   - "The receiver uses the lists to set the places where data will land", and the sender writes "as each part is
>     ready".
>   - "It asks no peer to choose a place or grant permission between steps."
>   - "We avoid choices and timing cases that would otherwise need checks while the work runs."
>   - Reuse of a page is told by "work sent back by our peers … We do not ask for a separate receipt."
>
> So memory is the abundant resource. The scarce ones are link time and step latency, and the ring spends the scarce to
> save the abundant.
>
> **What's stupid about "one wait per page":**
> - **The same event is signalled twice.** The bridge already learns every landing from the hardware, because a
>   receive's completion is the landing. The per-page word is the bridge announcing that same event again.
> - **Then every GPU thread that touches the page checks it,** with a system-scope fenced load: thousands of checks per
>   page of a fact the bridge already had.
> - **The check costs a large share of what it checks.** A 16 KiB page crosses a cable in about 1.7 µs at the recorded
>   9.3–9.5 GB/s, while the recorded fenced poll is 0.58 µs, a third of the transfer it guards.
> - **None of it is needed.** With fresh, non-overlapping pages for every execution, the plan already guarantees
>   nothing else writes those pages. A step needs one fact, that its inputs have landed, not one fact per page per
>   thread.

**The operator:**

> more broadly. how many of the algorithm choices that the user has firmly presented are compatible with very wide simd
> and very wide simt programming approaches? how many of the approaches has the user chosen are Actually Totally
> Parallel Safe if you do One Simple Thing that the user unquestionably expects the implementations to use, and perhaps
> has said to use explicitly many times? si there a class of waitles algorithms for limiting access only to buffers
> that are literally midwrite during a call? why are there 'pages' used here, metaphorically? i doubt it is to hide a
> struct inside each of them that causes global locks on repeated pointless reads...

**Claude:**

> **1. How compatible your choices are with very wide SIMD and SIMT.** Almost all of them are, and several exist to make
> that possible:
> - **Plans compiled ahead of time:** every kernel and transfer is fixed before the call, so nothing branches on data
>   at run time and every lane does the same thing.
> - **The page table, lowered to literal page-index reads and writes:** addressing becomes indexed gathers, with no
>   pointer chasing.
> - **Contiguous operands or paged views through a block table:** each thread finds its page with one lookup and reads
>   contiguously within it, the way PagedAttention does.
> - **Zero-copy views, and the mesh defining no arithmetic:** the kernels are the frameworks' own, written for SIMT
>   already.
> - **Your "lower to the literal machine: SIMD/SIMT, descalarize":** that's the instruction itself.
> - **No waits beyond true data dependencies, with spins only if bounded:** that fits SIMT as long as the check happens
>   once per step or per threadgroup, not once per thread.
> - **Every transfer scheduled together, relays as plain forwards, streamed partial results:** all static and uniform.
>
> What doesn't fit came from the agents, not from you:
> - per-page, per-thread fenced polls;
> - modulo rings that make one execution wait on an earlier one;
> - per-page signal words;
> - locks, leases and slots.
>
> **2. The one simple thing:** pages are bin-packed so that nothing live at the same time shares a page, including
> adjacent iterations of a repeating function. It's static single assignment at page granularity, decided by the
> compiler. You said it as "bin-pack over pages to ensure one-sided checkless waitless syncless writes, by ensuring
> nonoverlap of adjacent iterations".
> - With it, no write ever lands where someone else is reading or writing.
> - The write-after-read and write-after-write hazards disappear by construction.
> - The only hazard left is read-after-write: the program's own data edges, which the plan already holds and the
>   transport's completions already signal.
> - Under that one rule, every approach you've chosen is parallel-safe without any lock: one-sided writes, compiled
>   page lists, zero-copy views, relays, joint scheduling, streamed partials.
>
> **3. Is there a class of waitless algorithms that restricts only buffers being written mid-call?** Yes, in two
> families:
> - **Static: liveness analysis with interference.**
>   - Values live at the same time get different storage. That's register allocation applied to buffers: Chaitin's
>     graph colouring (1982), and Poletto and Sarkar's linear scan (1999).
>   - In ML runtimes it's offline memory planning: Pisarchyk and Lee's greedy-by-size arena planner (2020) and XLA's
>     buffer assignment.
>   - Nothing is checked at run time. A buffer being written is simply never anyone else's.
> - **Dynamic, with readers that never wait:**
>   - read-copy-update (McKenney and Slingwine, 1998), where writers build a new copy;
>   - double buffering;
>   - seqlocks (writers never wait, readers retry);
>   - epoch-based reclamation (Fraser, 2004).
> - **For consuming partial data:** a monotone landed prefix. One counter per range, which a consumer checks once
>   against a threshold, like Portals counting events or NCCL's step counters. Only data still in flight is held back,
>   and each step or threadgroup reads one word, not one per page.
>
> Your design is the static family plus landed prefixes. Nothing else is needed.
>
> **4. Why "pages."** As in virtual memory: a fixed-size unit with a page table that maps a name to a physical place.
> - **Placement and reuse without copies.** The indirection lets the compiler put any value anywhere in the region and
>   reuse memory by liveness, the way paging separates logical from physical memory.
> - **Agreement across nodes.** Every node knows ahead of time where each page lands, so both ends agree with no
>   negotiation.
> - **It matches the hardware:** registration in extents, transfers in whole units, SENDs and receives in 4 KB credit
>   frames.
> - **It keeps bin-packing simple,** because every unit is the same size.
>
> Pages are not there to hold a synchronization struct. Putting a readiness cell inside every page, re-read by every
> thread, turns the unit meant to remove coordination into a place where everyone contends.

**The operator:**

> "2. The one simple thing: pages are bin-packed so that nothing live at the same time shares a page, including
> adjacent iterations of a repeating function. " is this even *compatible* with modulo ring buffers attempting to
> invert control and hide a scheduler inside of a memory indexing system? or is this *strictly incompatible* with the
> full list of requirements?
> "- The write-after-read and write-after-write hazards disappear by construction.
> - The only hazard left is read-after-write: the program's own data edges, which the plan already holds and the
>   transport's completions already signal."
> is this solved by MASKING or is it solved by WAITING?
> finally, is the LIST OF MASKED PAGE INDICES a contiguous array iN ITS OWN SECTION OF THE MEMORY ALLOCATION, or is it a
> struct which MUST BE CONTIGUOSU WITH THE PAGE IT IS ABOUT FOR SOME STUPID REASON?

**Claude:**

> 1. **Strictly incompatible.** A modulo ring reuses the same `depth` pages every `depth` iterations. So within the
>    window, values live at the same time share pages by construction, and only a wait makes that safe. The
>    `t mod depth` arithmetic is a scheduler hidden in the address calculation: it decides when a page may be reused,
>    and it holds receives on earlier executions' landings and forwards. Memory indexing ends up dictating execution
>    order. Liveness bin-packing is a different allocator that assigns pages at compile time so nothing live shares a
>    page. You require one allocator, the page table and no rings, so the two can't coexist.
>
> 2. **Masking.** Each step reads only the pages its mask marks as valid: pages whose data has landed, read from the
>    landed list. In SIMT terms, lanes or threadgroups whose input pages aren't present are predicated off, and that
>    work runs in a later pass when they are. No thread spins. It's the form you described: "guards rather than
>    waits", and "function-local guards wrt data ordering or consumption patterns rather than using waits".
>
> 3. **A contiguous array in its own section of the allocation.** It's a dense list or bitmask of page indices sitting
>    beside the page table and page lists, not inside or next to the pages it describes. One vector load then gives a
>    whole group of lanes the state of many pages at once, and nothing sits on the data's cache lines. The
>    implementation that just landed does the opposite. Every page's landed word gets its own padded 256-byte cell at a
>    stride, `word + t × 256`. It isn't physically next to the page, but it is one struct per page, which every reader
>    has to load and fence individually. That's the per-page-struct pattern in all but placement.

**The operator:**

> cite all of hte literature and architectures which use the family of algorithms the user is unsubtly implying here.
> write up yet another change request document, just like the last one, which uses block quoted user and assistant
> turns for the body of the text to make it as hard as possible to pretend there's an ambiguity or decision to make
> here, rather than an unassailable and certain difference which must be remedied by a boring and canonical path

**Claude:**

> The family is four ideas used together: storage assigned ahead of time by liveness, so nothing live shares storage;
> read-after-write honoured by masks, not waits; masks and indices kept as dense arrays apart from the data they
> describe; and completion counted once per unit of work. The literature and the architectures, by idea:
>
> **Storage assigned by liveness, decided before the run**
> - G. J. Chaitin. Register allocation and spilling via graph coloring. SIGPLAN Symposium on Compiler Construction,
>   1982.
> - M. Poletto and V. Sarkar. Linear scan register allocation. ACM TOPLAS 21(5), 1999.
> - M. Lam. Software pipelining: an effective scheduling technique for VLIW machines. PLDI 1988. Its modulo variable
>   expansion gives each overlapping iteration its own copy, sized by lifetime at compile time, with no wait.
> - B. R. Rau. Iterative modulo scheduling. MICRO 1994. With Rau et al., The Cydra 5 departmental supercomputer, IEEE
>   Computer, 1989, and the Itanium (IA-64) rotating register files: hardware renaming per iteration, sized and proven
>   by the compiler.
> - Y. Pisarchyk and J. Lee. Efficient memory management for deep neural net inference. arXiv 2001.03288, 2020
>   (TensorFlow Lite's arena planner, greedy by size).
> - XLA's buffer assignment (TensorFlow XLA), and T. Chen et al., TVM: an automated end-to-end optimizing compiler for
>   deep learning, OSDI 2018 (static memory planning).
>
> **Read-after-write by predication, not waiting**
> - J. R. Allen, K. Kennedy, C. Porterfield and J. Warren. Conversion of control dependence to data dependence. POPL
>   1983 (if-conversion).
> - G. H. Barnes et al. The ILLIAC IV computer. IEEE Transactions on Computers C-17(8), 1968 (per-processor mode bits).
> - R. M. Russell. The CRAY-1 computer system. CACM 21(1), 1978 (the vector mask register).
> - W. D. Hillis. The Connection Machine. MIT Press, 1985 (per-processor context flags).
> - E. Lindholm et al. NVIDIA Tesla: a unified graphics and computing architecture. IEEE Micro 28(2), 2008 (SIMT,
>   divergence by active masks).
> - Intel AVX-512 opmask registers k0–k7 (Intel SDM); N. Stephens et al., The ARM Scalable Vector Extension, IEEE Micro
>   37(2), 2017 (predicate registers); the RISC-V Vector Extension 1.0 (the mask register v0).
>
> **Masks and indices as dense arrays apart from the data**
> - T. Kilburn et al. One-level storage system. IRE Transactions on Electronic Computers, 1962 (Atlas, the first page
>   table); P. J. Denning. Virtual memory. Computing Surveys 2(3), 1970.
> - W. Kwon et al. Efficient memory management for large language model serving with PagedAttention. SOSP 2023 (the
>   block table, a dense array separate from the KV pages).
> - Apache Arrow's columnar format: the validity bitmap is its own buffer, beside the values buffer.
> - Y. Saad. Iterative Methods for Sparse Linear Systems, 2nd ed., SIAM 2003 (CSR: row pointers and column indices as
>   arrays apart from the values).
> - G. E. Blelloch. Vector Models for Data-Parallel Computing. MIT Press, 1990, and Prefix sums and their applications,
>   1990 (compaction of a mask into a dense index list by scan).
>
> **Completion counted once per unit of work**
> - The Portals 4 network programming interface (Sandia): counting events, one threshold per operation set.
> - OpenSHMEM 1.5: put-with-signal (`SHMEM_SIGNAL_ADD`, `shmem_signal_wait_until`), and NVSHMEM's equivalents.
> - NVIDIA PTX ISA: `mbarrier` with expected transaction bytes for bulk copies, one barrier phase per tile.
> - R. Belli and T. Hoefler. Notified access: extending remote memory access programming models for producer-consumer
>   synchronization. IPDPS 2015.
>
> **Readers that never wait, for contrast**
> - M. Herlihy. Wait-free synchronization. ACM TOPLAS 13(1), 1991.
> - P. E. McKenney and J. D. Slingwine. Read-copy update. 1998.

## State of the code

- mesh `rdma/mesh.h`:
  - A plan runs hdr `executions` times, its pages "ringing through hdr `depth` slots". The list is `hops = depth × n`
    entries, slot s's n at `[s n, s n + n)`, and execution t uses slot `t mod depth`'s page.
  - A hop's receive of execution t is posted only once its receives of executions 0 .. t − (depth − 1) have landed,
    and, from t = depth, once its cell of execution t − depth reads its `forwards` at offset 8.
  - Every word has one 256-byte cell per execution, at region offset `word + t × hop_stride` (`hop_stride` 256). A
    landed word stands for one page, and its cell holds that page's sent count at offset 8.
- mesh `rdma/mesh-flow.c`: `plan_shape` (`executions`, `depth`, the slot), `struct flow_hop`
  `{word, position, through, forwards}`, and the receive window and forwards check before a receive is posted.
- mesh `rdma/mesh-collective.c`: `mesh_plan` writes the hops of every slot and a relay's pages for each slot.
- mesh `rdma/mesh-metal.m`: `struct mesh_metal_input` `{completion, offset, stride}`, one input per received page; the
  GPU waits on each page's own landed word.
- metal-microbench `metal_recording.m`: the waits spliced into recorded kernels (`mesh_read_N`, line 1880), one per
  range per thread, where a range is now one page.
- metal-microbench `tools/e2b_replay.m`: page-sized ranges.

## Scope

- mesh `rdma/mesh.h`: `depth`, `hop_stride`, the slot layout of the hop list, and the per-word, per-execution cells.
- mesh `rdma/mesh-flow.c`: `plan_shape`, `struct flow_hop`'s `forwards`, the receive window, and the forwards check.
- mesh `rdma/mesh-collective.c`: `mesh_plan`'s slots and per-slot relay pages, and its page assignment.
- mesh `rdma/mesh-metal.m`: `struct mesh_metal_input` and the GPU's per-page waits.
- metal-microbench `metal_recording.m`: `mesh_read_N`'s per-range waits.
- metal-microbench `tools/e2b_replay.m`: its page-sized ranges.

## Claude's proposal: the canonical path

This section is Claude's. With it, this document is the unit of work.

1. **One allocator, by liveness, at compile time.** `mesh_plan` assigns pages so no two values live at the same time
   share a page, including the same value in adjacent executions (the static family above: interference by liveness,
   greedy by size). The ring goes: `depth`, `t mod depth`, the slot layout, the receive window, and the forwards gate.
   A relay's page is a page like any other, assigned by liveness.
2. **Read-after-write by mask.** Each node's landings go into one contiguous list in its own section of the region,
   beside the page table and the page lists: the pages of each step that have landed, as a dense bitmask or index list.
   A step reads its mask with vector loads, and lanes or threadgroups whose input pages haven't landed are predicated
   off and run on a later pass. Nothing spins on a page.
3. **No per-page cells.** The 256-byte cell per word per execution, the landed word per page and its offset-8 sent
   count go. A supplied word stays what releases a send, and one store still releases a range.
4. **The GPU reads the mask, not a word per page.** `struct mesh_metal_input` and `mesh_read_N` read the step's mask
   from the landed list instead of fencing one word per page per thread.
5. **What judges it:**
   - One configuration runs in a runtime, and no call site handles both forms.
   - The document's builds are clean: `make -C rdma` and `make -C rdma native-audit` in mesh, and `make
     .build/e2b_replay` in metal-microbench.
   - Parity is shown by the normal program, when the operator authorizes runs.
   - Every wait that remains is bounded, and every failure is an error value.
   - Commits are pushed to main as each piece builds.

## Appendix A: the allocator

This appendix is Claude's. It defines the one allocator this change requires, and cites both the strategy in the code
now and the strategy that replaces it to their closest literature.

### A1. Where pages are allocated now (mesh 7e9936d5)

- **First fit with compare-and-swap, wherever a caller needs pages.** `rdma/mesh-dataflow.c` `mesh_rows_alloc` and
  `mesh_arena_alloc` do "one first-fit scan over an explicit pair of bitmaps", and take each row's owner word by
  compare-and-swap. They are called from `rdma/mesh-call.c` (a client's sections, lines 22, 32 and 62) and from
  `rdma/mesh-records.c` (the bridge's record pages, line 76).
- **Ring slots in the plan.** `rdma/mesh-collective.c` `plan_pages` lays each section out as `depth` slots of its
  pages, and `plan_row` gives slot s's page k as `first + slot × (pages / block) + k`. Execution t uses slot
  `t mod depth`. Reuse of a slot is gated while the plan runs: `rdma/mesh-flow.c` posts a receive of execution t only
  once executions up to t − (depth − 1) have landed and, from t = depth, once execution t − depth's forwards are
  counted.

### A2. The current strategies and their closest literature

- **First fit over a bitmap, at the call that needs memory.** This is run-time dynamic storage allocation: D. E. Knuth,
  The Art of Computer Programming, vol. 1, §2.5 (1968), and P. R. Wilson, M. S. Johnstone, M. Neely and D. Boles,
  Dynamic storage allocation: a survey and critical review, IWMM 1995. It optimizes for allocating without knowing the
  future: each request is served from what is free at that moment. It knows nothing of a value's lifetime, and it
  needs an atomic claim because other callers allocate concurrently.
- **Ring slots reused under a run-time window.** This is the bounded buffer: E. W. Dijkstra, Cooperating sequential
  processes, 1965. Its reuse gate is credit-based flow control: H. T. Kung and R. Morris, Credit-based flow control for
  ATM networks, IEEE Network 9(2), 1995. In GPU collectives it is NCCL's per-connection FIFO of `NCCL_STEPS` slots with
  head and tail counters. It optimizes for bounded memory, and it pays with a wait before a slot is reused.

Neither knows the program's dependences, and both decide memory while the program runs.

### A3. The allocator this change requires

**What it is.** One function, called once per plan, inside `mesh_plan`, before the plan's first execution. It is the
only allocator of a program's pages. Nothing allocates a program page while the program runs: not the bridge, not a
call site, not a GPU kernel.

**Its inputs.**
- Every value instance the plan moves or computes, across all of the plan's executions t = 0 … executions − 1, with its
  size in pages.
- The plan's dependence graph: each value instance's producer and its consumers, the program's own data edges, and the
  transport's edges (a SEND before its landing, a landing before its forward).

**Interference, by happens-before, not by time.** Two value instances interfere unless the dependence graph orders
every read of one before the first write of the other (L. Lamport, Time, clocks, and the ordering of events in a
distributed system, CACM 21(7), 1978). Only values that don't interfere may share a page. A shared page then needs no
wait, because the program's own edges already order its last read before its next write. This is how CUDA graph memory
nodes reuse physical memory: only where the graph's topology guarantees the lifetimes don't overlap (NVIDIA, CUDA C++
Programming Guide, "Graph Memory Nodes").

**The assignment.**
- A page is the unit, so every value instance needs whole pages. Ordered along a topological order of the dependence
  graph, the instances' lifetimes are intervals. Colouring an interval graph greedily by start point, the left-edge
  algorithm, uses exactly as many colours as the largest set of intervals live at once, which is optimal (T. Hashimoto
  and J. Stevens, Wire routing by optimizing channel assignment within large apertures, DAC 1971; M. C. Golumbic,
  Algorithmic Graph Theory and Perfect Graphs, 1980). It is linear scan applied to pages (M. Poletto and V. Sarkar,
  Linear scan register allocation, ACM TOPLAS 21(5), 1999).
- A value instance of several pages takes a run of free colours at its start point. Where sizes differ, this is the
  offline dynamic storage allocation problem (M. R. Garey and D. S. Johnson, Computers and Intractability, 1979;
  J. Gergov, Algorithms for compile-time memory optimization, SODA 1999; A. L. Buchsbaum et al., OPT versus LOAD in
  dynamic storage allocation, STOC 2003), solved in the ML compilers by greedy by size: Y. Pisarchyk and J. Lee,
  Efficient memory management for deep neural net inference, arXiv 2001.03288, 2020 (TensorFlow Lite's arena planner);
  XLA's buffer assignment and heap simulator; T. Chen et al., TVM, OSDI 2018 (static memory planning).
- **Repeated executions are unrolled, not ringed.** The same value in executions t and t + 1 are two instances. They
  share a page only if the dependence graph already orders t's last read before t + 1's first write. Otherwise they
  get different pages, as compile-time modulo variable expansion gives each overlapping iteration its own copy (M. Lam,
  Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo scheduling, MICRO 1994, and the Cydra 5 and IA-64
  rotating register files). The number of copies is decided by the allocator from the graph. Nothing counts or waits
  for it at run time.
- **Memory is the abundant resource.** A Studio registers 82 of its 96 GiB. Minimizing peak pages is the allocator's
  tie-breaker, never a reason to add a wait.

**Its output.** Every value instance's pages, written into the page table and the page lists before the first
execution. The landed list's section is laid out beside them (proposal item 2).

**Its failure.** If the plan needs more pages than its section holds, `mesh_plan` fails with `ENOMEM` before anything
runs. There is no fallback allocation at run time.

**Sharing the region between programs.** Each program claims its section of the region once, when it attaches,
and the allocator assigns pages only inside that section. Two programs never allocate from each other's section. The
bridge's own pages (its records) are laid out once with the region, not taken from any program's section.

### A4. What goes

- `mesh_rows_alloc` and `mesh_arena_alloc` as call-site allocators: `mesh-call.c`'s three calls and
  `mesh-records.c`'s call. One claim of a program's section at attach remains.
- `plan_pages`' `depth` slots and `plan_row`'s slot arithmetic.
- `mesh-flow.c`'s receive window and its forwards gate, and the hdr fields `depth` and `hop_stride` that serve them.
