# Change request: pages assigned by liveness, landings read by mask from a contiguous list

The operator's and Claude's turns of 2026-11-03 on the page readiness word and the plan's ring, verbatim; the
literature; the state of the code they change (mesh 7e9936d5, metal-microbench f154814d); the data structures and
functions in scope; and the canonical path.

Appendix C holds the operator's rulings on the first implementation and Claude's consolidated revision (C3); where C3
differs from the canonical path, Appendix A3 or Appendix B, C3 holds. Appendix D holds the operator's rulings on the
revision and Claude's proposal (D.3), which is the unit of work; where D.3 differs from C3 or anything earlier, D.3
holds.

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

This section is Claude's. Appendix C3 consolidates the review of its first implementation and holds where they differ.

1. **One allocator, at compile time, as Appendix A defines it and the operator ruled (Appendix B).** `mesh_plan`
   unrolls the plan's executions into value instances and assigns their pages once, before the first execution.
   Pages are reused only where the plan's dependence graph orders the last read before the next write, as CUDA graph
   memory nodes reuse memory where the graph's topology guarantees it. The ring goes: `depth`, `t mod depth`, the
   slot layout, the receive window, and the forwards gate. A relay's page is a page like any other.
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

## Appendix B: the operator's ruling on the allocator

**The operator:**

> "This is how CUDA graph memory nodes reuse memory." this + unrolling is the user's proposal for how all of this is
> implemented for the subsidiary component of this change request. okay. append the relevant proposals to the change
> request document(s) and we are ready to have a subagent implement them.

So the allocator is implemented as Appendix A's A3 states it: the plan's executions unrolled into value instances, and
pages reused only where the dependence graph's topology guarantees the lifetimes don't overlap, as CUDA graph memory
nodes do. No other allocation strategy is in scope.

## Appendix C: the operator's rulings on the first implementation, and the consolidated revision

The implementation of this request (mesh 54070db1, metal-microbench 472390a8) and its review on 2026-11-03: Claude's
report of where it departs, and the operator's and Claude's turns that followed, verbatim (C1); where this document's
own text produced the departures (C2); Claude's proposal consolidating the review (C3); and what goes (C4). C3 is the
unit of work. Where it differs from the canonical path, Appendix A3 or Appendix B, C3 holds.

### C1. The turns

In order. The turns name the proposal's items C3.1 to C3.7; C3 keeps those numbers, and its text supersedes what the
turns quote of it.

**Claude** (the implementation report):

> **Where it departs from what you required, worst first:**
> 1. **Read-after-write is a wait, not a mask.** Your answer was masking: lanes whose pages haven't landed are predicated
>    off and run in a later pass. Instead, each lane spins on its step's mask, up to 2^23 polls (about 4.9 s). The
>    coordinator's reason is that the decode is one command buffer, so a later dispatch would run immediately. It's
>    bounded and it checks one mask, not a word per page, but it is a wait, and it happens per lane rather than per
>    threadgroup.
> 2. **The allocator doesn't place everything.** Weights, KV cache, pooled intermediates and the GPU's publication
>    records are laid out when they're created, in the program's claimed run. That leaves two allocation paths, not one.
> 3. **New coordination mechanisms were added:**
>    - an "erased" plan flag with a Dekker-style fence pair against the ports;
>    - a bounded 5 s wait at attach for a retired client's claim;
>    - a bound mark the host reads after the run.
> 4. **One client holds the region at a time.** At attach the claim takes everything that's free. That keeps 10-01's
>    single-client region, against your rule that every bridge serves many clients at once.
> 5. **Nothing is reused across executions yet.** `e2b_replay.m` declares no data edges, so every step's values get fresh
>    pages. That's correct, given memory is the abundant resource, but the reuse your ruling describes isn't exercised.
>
> **Still open:**
> - a narrow race in which a bridge cancel hits one step of a just-written plan;
> - colouring becomes quadratic if a program declares dependences across executions (none does today);
> - the host's final completion wait in `e2b_replay.m` is unbounded, as it was before this change.

**Claude** (on which programs declare such dependences):

> **Why passing the edges would be slow.** The colouring has one item per value per step of the run (items = values ×
> steps). Each item scans the runs already placed and calls `plan_interferes`, which loops over the value's reads. That
> is O(items²) times the reads per value. The coordinator's own estimate is over ~8 s at e2b's largest sizes.
>
> The cost comes from unrolling over the run's length. Modulo variable expansion (Lam 1988; Rau 1994, the source the
> change request cites for unrolling) unrolls by the longest lifetime counted in iterations, not by the trip count. A
> value live for k iterations gets k copies, which rotate. A decode exchange is dead before the same exchange in the
> next step is written, so the factor is 1 or 2. The item count would then be values × 2 whatever the step count. The
> plan also wouldn't need the run's length: the last three commits (24ec8cda, cfada099, 472390a8), which refuse runs
> whose steps differ from the plan's executions, follow from this choice.

**The operator:**

> 1: what's quadratic here really
> 2: is the most appropriate algorithm for this topic quadratic in anything (flops, memory alloc, memory bandwidth)? at
> runtime? compile time?
> 3: how much unrolling is really required by the circumstances?
> "4. One client holds the region at a time. A"
> this parses to me as nonsense. don't make any 'grabby' code which transitively or terminally 'grabs' resources it is
> not using; there is no benefit to doing this prematurely and it doesn't make sense conceptually either.
> "2. The allocator doesn't place everything. "
> irresponsible and nonsensical.
> "1. Read-after-write is a wait, not a mask. Your answer was masking: " then the change request document's
> requirements simply have not been satisfied yet. revision is required.
> "5. Nothing is reused across executions yet. e2b_replay.m declares no data edges, so every step's values get fresh
> pages. That's correct, g"
> then your code isn't correct for deliberately avoiding demonstrating the requirements of the change request.
> "- a narrow race in which a bridge cancel hits one step of a just-written plan;"
> the races and unbounded waits must be addressed; no detail of how or why is required.
> " unrolls by the longest lifetime counted in iterations, not by the trip count."
> so what i'm hearing is that there is a reference algorihtm which should have been cited and used ,but someone wrote
> an incomplete or incorrect variant algorithm instead. choose any sensible approach from the literature. i do not need
> to hear about partial or incomplete code which does not come from the specificatino and does not come from the
> literature either.

**Claude:**

> **1. What's quadratic.** Everything quadratic is compile-time work inside `mesh_plan`, in two places:
> - **The colouring** makes one item per value per step. Each item is tested against the runs already placed below
>   where it lands, so there are up to items² calls of `plan_interferes`, and each call loops over the value's reads.
>   Items = values × steps.
> - **The happens-before table** holds reads × values × ⌈steps/64⌉ words: quadratic in one step's values, linear in
>   steps.
>
> What's really being squared is the run's length. The step count is a runtime argument (`e2b_replay`'s fourth, 16 by
> default), and it enters compile time squared because each step's copy of a value is its own item. One step's values
> are a small fixed set, and their square is harmless. Today none of the quadratic code runs: with no edges declared,
> the colouring takes its every-pair-interferes branch. That branch is linear in time, but it lays every (value, step)
> end to end, so memory grows with steps instead.
>
> **2. The appropriate algorithm.**
> - **Runtime: nothing is quadratic**, and nothing is even linear beyond the data itself. There are no allocations, no
>   bookkeeping flops and no extra bandwidth. Every address is a constant written into the plan before the first
>   execution. With u copies, execution t runs copy (t mod u)'s page list. Lam unrolls the code; Rau's rotating register
>   files do the same thing with a base register. The landed list gets one record per landed page, written once and read
>   once, which is the same order as the data.
> - **Compile time: quadratic in one execution's graph, once per plan.**
>   - Packing values of different sizes optimally is NP-hard (offline dynamic storage allocation; Garey and Johnson), so
>     every compiler uses a heuristic.
>   - Greedy by size (TensorFlow Lite's planner, from Pisarchyk and Lee; XLA's heap simulator) is quadratic in the
>     number of values in its worst case.
>   - Left-edge would be n log n, but it needs lifetimes that are intervals on one total order. A page written by
>     another node's send doesn't have one.
>   - The happens-before table is quadratic in one execution's values.
>   - With values × u items, that's hundreds to a few thousand items for a decode step: at most about 10^6 interference
>     tests. It never depends on the run's length.
>
> **3. How much unrolling.** For each value, take the smallest lag L at which every read of it in step t happens before
> its write in step t + L. The unroll factor u is the largest such L. This is Lam's rule (the longest lifetime over the
> initiation interval), with lifetime measured by happens-before because nothing here has a fixed cycle count.
> - **Tensor-parallel decode (E2B, Megatron layers): u = 1, so no unrolling.** Every exchange involves every rank, and
>   each rank's queue runs in order. Rank m sends its part of exchange k in step t + 1 only after reading the exchange
>   before it. That exchange held rank n's part, which n produced after reading exchange k of step t. So every step
>   reuses the same pages.
>   - Within a step, exchanges k and k + 1 interfere, because m can send k + 1 before n has read k. Exchanges k and
>     k + 2 don't.
>   - So the colouring gives double buffering by itself. The decode's exchange memory is about two exchanges' worth,
>     however many steps run.
> - **Recurrent state updated in place** (an LSTM tick that reads h_t while it writes h_{t+1}): u = 2.
> - **1F1B pipeline parallelism:** u is up to the number of stages, which is the in-flight micro-batch count 1F1B is
>   known for (D. Narayanan et al., PipeDream, SOSP 2019).
> - **A value whose reads nothing ever answers** has no bound, so no unroll factor is right. The plan fails with
>   `ENOMEM` and names the value.
>
> The run's length is never the unroll factor.

**The operator:**

> do you think item 3 is endorsed by any particular algorithm? is it waitless, zero copy, one-sided-writing, etc?

**Claude:**

> No. None of the three comes from the specification, and none comes from an algorithm in the literature as it's used
> here. All three are waits, and all three need both sides to take part. Zero-copy doesn't apply to any of them, since
> they're control words, not data. Each covers a place where something still live is overwritten, or where memory is
> held that nothing uses.
>
> - **The erased flag and its Dekker fences** (`mesh_plan_erased`, mesh-dataflow.c). A new plan zeroes and rewrites the
>   previous plan's supplied words and landed records in place, because they sit at fixed offsets in the layout. Before
>   doing that, it stores `hops = 0`, issues a seq_cst fence, and spins for up to MESH_PLATFORM_NS (5 s) until no
>   port's `prepared` still holds the old generation. Each port fences on its side too.
>   - The fence pair is a correct store–fence–load primitive (Dekker, in E. W. Dijkstra, Cooperating sequential
>     processes, 1965).
>   - What it builds is RCU's grace-period wait without RCU's copy (P. E. McKenney and J. D. Slingwine, Read-copy
>     update, PDCS 1998). RCU publishes the new version to new memory and reclaims the old one later. It never waits
>     so that it can overwrite in place.
>   - It also creates the cancel race: a late `plan_cancel` lands in records the new plan has just zeroed.
>   - Once the allocator places the records (C3.2) and a program holds only its plan's pages (C3.4), a new plan's
>     records are new pages. The bridge's own turn releases the old ones once no port holds them, which is deferred
>     reclamation (RCU's call_rcu; K. Fraser, Practical lock-freedom, 2004). Then there is no wait, no fence pair and
>     no race.
> - **The 5 s wait at attach.** A new client spins until the ports drop a retired client's plan, releases that claim,
>   then takes everything free. No algorithm calls for this. It is the grab waiting to reclaim memory so it can grab
>   again. With C3.4 a client takes only what its plan uses, from what is free, so the wait has nothing to do.
> - **The bound mark.** A lane that reaches its 2^23-poll bound writes `UINT64_MAX` into the step's landed record, and
>   the host reads it after the run. That record then has five writers: landings, `link_close`, `plan_cancel`, a GPU
>   lane and the program's stop. One-sided protocols keep each landing word to one writer, the transport (NCCL's LL
>   flags; L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986, single-writer registers). The
>   mark exists only because lanes spin (report item 1). With masking, no lane has a bound to reach.
>
> Applying the same test to C3.1: it isn't waitless either. It has one wait per pass, held by the command queue. In a
> chain of dependent steps like decode, some ordering point is unavoidable. Its wording also gave the landed count two
> writers ("the client raises it at the step's deadline"). Fixed: the bridge alone writes the landed list and its
> count, and the client alone signals the queue's event, when the count is complete or at the deadline.

**The operator:**

> "s: a new plan zeroes and rewrites the previous plan's supplied words and " why is this necessary and how much
> bandwidth does this wate? algorithms review please? data model? what is the hardware like at the level of registers
> and superscalar executions of operations we can load widely and implicitly literally concurrently? command buffers?

**Claude:**

> It isn't necessary. The zeroing exists because of two choices. First, the plan's words record absolute state
> ("0 = not yet") in areas that get reused in place. Second, it tidies bytes that no computation reads.
> `mesh_plan_erased` (the 5 s fence-and-spin) exists only so this zeroing doesn't race ports that are still reading
> the old words. Drop the zeroing and the erase has nothing left to protect.
>
> **What gets zeroed.** This is mesh-collective.c:749–763. It runs on every `mesh_plan`, on one CPU thread, before
> `hops` is stored.
> - **Every declared page of every execution** (the `memset` at line 757).
>   - **The cost:** one extra write of the node's whole routed data for the run, so 16 steps means 16 steps' worth of
>     pages. Every zeroed byte is then overwritten by its landing or its producer, so these are dead stores.
>   - **Reuse saves nothing:** a page two instances share is zeroed once for each, as the code's own comment says.
>   - **Where it came from:** e2b_replay.m used to do it, for two reasons. One was to take first-touch page faults
>     early, but the region is now wired once at registration (data-model §2). The other was to make "an instance's
>     untouched tail deterministic": the bytes past a value's end in its last page, which no kernel reads.
>   - **Why it moved into `mesh_plan`:** the hardware holds a SEND in credit until its receive is posted (§3), so
>     zeroing after the plan would erase data that had already landed.
> - **The supplied words and landed records.** That's releases × executions × 8 B, plus executions × the record
>   stride: kilobytes. But they're written one 8-byte atomic store at a time, the narrowest store the core has. A plain
>   fill moves 32 B per `stp`, or a whole cache block per `dc zva`.
>
> There's no per-step bandwidth cost. Each plan writes the run's routed bytes once more, one thread, before the first
> step can start.
>
> **Algorithms review.** A word needs a reset only if it records absolute state. The literature avoids that in four
> ways, and none of them resets anything:
> - **Sequence or epoch values** (NCCL's LL/LL128 flags carry the step; NVSHMEM signals rise and are waited on with
>   ≥). A stale word never matches the current epoch. This fits the supplied words, using the plan's generation and
>   the execution number packed so the value only rises. It also fits the cancel word: "cancelled from execution t",
>   compared rather than set to `UINT64_MAX`.
> - **Sense reversal, or phase parity** (Mellor-Crummey & Scott 1991; PTX mbarrier's phase bit). A landing XORs its
>   bit instead of ORing it, and the expected sense flips each time a record copy is reused.
>   - The landed list stays one dense bit per page.
>   - A copy is set to its first sense only once, when the allocator places it: u records, not one per execution.
>   - It's safe because the allocator reuses a copy only where the last read happens before the next landing.
> - **Counted completion with rising thresholds** (Portals 4 counting events; `MTLSharedEvent`'s 64-bit value, which
>   only rises). Execution t waits for the count to reach its own target.
> - **Data pages need nothing.** Every read in a placed value follows a write of the same bytes, and a mask never
>   reads outside its page (§6).
>
> **Data model.** §5 already rules this out. "Once the plan is set, new input values can follow the same writes."
> Peers' returned work "already tells us what we need to know. We do not ask for a separate receipt." A per-plan reset
> is a receipt in another form.
>
> **The hardware at register level.**
> - **GPU.**
>   - A SIMD-group is 32 lanes in lockstep, with a per-lane execution mask, which is what predication uses.
>   - With 16-byte vector loads per lane, one coalesced load reads 512 B: the landed bits of 4,096 pages. Testing a
>     lane's pages is then an AND or XOR in registers.
>   - `simd_ballot` turns 32 predicates into one mask. `simd_prefix_exclusive_sum` compacts the not-yet-landed work
>     into a dense list (Blelloch's scan, already cited in the request).
>   - Kernels stream at about 620 GB/s of the 1.2 TB/s (data-model §1).
> - **CPU** (bridge and planner). 128-bit NEON registers, several load/store pipes, 32 B per `ldp`/`stp`. Plain fills
>   vectorize; loops of `_Atomic` stores don't.
> - **Command buffers.** Within one command buffer, Metal already provides masking and its later pass without any
>   lane waiting:
>   - `dispatchThreadgroups(indirectBuffer:)` reads its threadgroup count from GPU memory when the dispatch starts.
>     The masked pass writes the later pass's size, which is zero when everything has landed.
>   - `encodeWaitForEvent:value:` holds back the encoders after it in the same command buffer until the bridge's count
>     reaches its value. The value only rises and is never reset.
>   - That removes the coordinator's reason for spinning: it said that with every step in one command buffer, a later
>     dispatch would run immediately.
>   - Concurrent compute encoders with explicit memory barriers let independent work fill the gap.

**The operator:**

> "- Data pages need nothing. " it is this specific requirement which should draw more attention from you. i would
> suggest the ptx mbarrier phase bit and similar; if you want to rotate through an int2 of phases instead of the
> reductive int1 (or bit) go for it, idc.
> "- Every declared page of every execution "
> yeah thats too much. i could tolerate a waste of bandwidth for absolute state tracking pages or indexings within them
> but the other algorithsm review tells us we don't even *need* that. so we need to get rid of this entire class of
> waste.
> "- Command buffers. Within one command buffer, Metal already provides masking and its later pass without any lane
> waiting:"
> so all of the remarks the user wrote about masking and paging were foreshadowing the literal hardware's literal abi
> and literal properties. the situation is worse than you realized: the session was with a world-class simd programmer
> this entire time.
> "Two things I haven't verified:
> - Apple's per-lane load width. " idk compile time constant or pick something arbitrary and nonmaladaptive. i don't
> care about this in fine grained detail rn compared to wasting bandwidth on every page use on prealgorithmic hunches.
> "- Whether any kernel reads a value's tail bytes. "
> what's a tail bytes? i don't know how to use a computer iwht less than 16 registers and less than 64 bits of
> concurrent reads per cycle because i wasn't born in 1930. review the data model etc.

**Claude:**

> Yes. Pages, masks, a dense list kept apart from the data, compile-time placement and phases are the hardware's own
> terms: execution masks, wide coalesced loads, indirect dispatch, events whose values only rise. My earlier documents
> treated them as preferences to argue with.
>
> **What a "tail" is.** It's a scalar-loop word: the bytes left over when a value's length isn't a multiple of the
> loop's step. I meant the bytes between a value's last element and the end of its last page. The data model has no
> such thing:
> - **A page is the unit.** "A mask picks values within one page… It cannot name a place outside that page" (§6).
> - **Every load is a whole vector inside a page,** and the mask decides which lanes count. Bytes no mask picks are
>   never data, whatever they hold.
> - **The guarding happens in registers, not memory.**
>   - Apply a mask with select, never by multiplying by 0 or 1, since 0 × NaN is NaN.
>   - A reduction puts its identity into masked-off lanes: `select(identity, x, m)`.
>   - Hopper's TMA does this in hardware: it fills out-of-bounds tile elements with zero as it loads, and nothing
>     zeroes memory.
>
> So the "unverified" item was the scalar hunch itself, and I've dropped it. Faulting pages in is the region's job,
> done once when it's wired (§2), never a program's `memset`.
>
> **Data pages need nothing.** A data page holds payload only. Its producer or its landing writes each byte once per
> use, consumers read it, and nothing else touches it: no header, no flag, no zeroing. All state lives in the landed
> list's own section. That's the floor for traffic: one write per byte, plus its reads.
>
> **The phases** (mbarrier's phase bit, widened to two):
> - **Encoding:** each page gets a 2-bit phase in the dense landed list, Gray-coded 00 → 01 → 11 → 10. Each step
>   flips exactly one bit, so a landing is one atomic XOR of one bit. Unlike an add, it never carries into the next
>   page's field.
> - **Reading:** a reader of execution t compares the field with the phase t expects. Equal means landed; one behind
>   means not yet. Anything else means a write ran ahead of the allocator's ordering, which a 1-bit parity can't
>   detect.
> - **Starting:** fields start at 00 when the allocator places the list, once per plan, the way `mbarrier.init` runs
>   once per kernel. That's 2 bits per page, never per execution, and never in a data page. Supplied and cancel words
>   rise with the executions and start the same way.
> - **Loads:** each lane loads `MESH_LANE_BYTES` = 16 bytes of the list, a compile-time constant, which covers 64
>   pages' phases.
> - **The landed count** behind the queue's event only rises.
>
> **The rest of this waste in the serving code** (C4 already lists mesh-collective.c:757 and 762–763):
> - e2b_replay.m:627 zeroes every local operand's whole extent, for first-touch faults and the tail.
> - e2b_replay.m:840 zeroes every pooled intermediate.
> - metal_recording.m:1546 zeroes the projection kernels' partial sums, which every threadgroup writes before anything
>   reads them. The arrival counters after them already work like an mbarrier: the last arrival restores the count
>   (native_projection.metal:232), the way phase completion reloads it.
>
> Only mesh-link.c:245 stays: it zeroes the region's header once, when the bridge creates the region.

**The operator:**

> update the appendix and write a proposal consolidating all of this review.

### C2. Where this document produced the departures

- A3, "Its inputs": "Every value instance the plan moves or computes, across all of the plan's executions t = 0 …
  executions − 1". Appendix B: "the plan's executions unrolled into value instances". Both unroll by the trip count.
  The modulo variable expansion A3 cites unrolls by the longest lifetime.
- A3, "Sharing the region between programs": "Each program claims its section of the region once, when it attaches".
  No plan exists at attach, so no size is known, and the implementation's claim took everything free.
- A3, "The assignment": "Ordered along a topological order of the dependence graph, the instances' lifetimes are
  intervals", and left-edge "is optimal". A landing's write is set by another node's send, so under happens-before the
  lifetimes are not intervals on one order. The implementation tests happens-before and colours greedily by size,
  which is right.
- Report item 5 called fresh pages for every step "correct". The operator's ruling: the code isn't correct when it
  avoids demonstrating the requirements.
- Report item 3's mechanisms come from neither this request nor the literature as they are used (C1, the operator's
  question on item 3 and Claude's answer), and they go (C4).

### C3. Claude's proposal: the consolidated revision

This section is Claude's. It consolidates the review in C1 into one proposal. Its items keep the numbers the turns use.

**The model.** Four rules hold everything below together:
- **A data page holds payload only.** Its producer or its landing writes each byte once per use, its consumers read
  it, and nothing else touches it: no header, no flag, no zeroing, no tail. A mask picks values within one page
  (data-model §6), and masked-off lanes are guarded in registers.
- **State lives apart from the data,** in the landed list's own section. Each state word has one writer, and no state
  word is reset while its plan stands.
- **Everything is placed once, at compile time,** by one allocator, and a program holds exactly what its plan uses.
- **No lane waits.** Read-after-write is a mask and a later pass. The only waits left are one per pass, held by the
  command queue, and each ends in bounded time.

1. **Read-after-write by mask and a later pass.**
   - A step's first pass loads its receives' phases from the landed list, `MESH_LANE_BYTES` (a compile-time 16) per
     lane, and computes the lanes whose pages have landed. The rest are predicated off. A mask is applied by select,
     never by multiplying by 0 or 1 (0 × NaN is NaN), and a reduction puts its identity into masked-off lanes.
   - The first pass compacts the work it left (`simd_ballot`, `simd_prefix_exclusive_sum`; G. E. Blelloch, Vector
     models for data-parallel computing, 1990) and writes the later pass's threadgroup count.
     `dispatchThreadgroups(indirectBuffer:)` reads that count when the later pass starts, so the later pass is empty
     when everything has landed.
   - The later pass is held at the queue by `encodeWaitForEvent:value:` until the node's landed count reaches
     execution t's target, (t + 1) × the node's receives per execution. This is counted completion: Portals 4
     counting events (R. Brightwell, B. Barrett et al., The Portals 4.0 network programming interface,
     SAND2012-10087), which count successes and failures apart; CUDA's `cuStreamWaitValue32`; Metal's
     `MTLSharedEvent`. The event's value only rises: execution t's later pass waits for t + 1.
   - Landings and cancellations both arrive on the count, as arrivals do on an mbarrier. A cancelled receive arrives
     without changing its phase, so the later pass finds it one phase behind. Its lanes end as an error that the later
     pass writes into the step's status, a word only the GPU writes, which the host reads after the run. No cancel
     word is needed. A link loss is such a cancellation.
   - The client alone signals the event: when the count reaches the step's target, or at the step's deadline. Its host
     thread waits on the count's address with a timeout (`os_sync_wait_on_address`, woken by the bridge's arrivals),
     so that wait is bounded too.
   - Both lane waits in `mesh_read_N` go: the remote wait on the step's mask, and the local countdown. A producer on
     the same queue comes before its consumer by the command buffer's order or a memory barrier. A host producer is
     ordered by the event.
   - This is not waitless. There is one wait per pass, held by the queue, and no lane waits. Concurrent compute
     encoders with memory barriers let independent work run in the gap.
2. **The allocator places every value.** Weights, the KV cache, pooled intermediates, exchanges, the GPU's publication
   records, the landed list and the supplied words are all placed by `mesh_plan`'s allocator, in the pages it computes.
   Values written before the first execution and read by every execution (weights; the KV cache as one array) are
   placed once, for the plan's whole life, as Lam places loop invariants outside the kernel. Nothing is laid out when
   it's created, and no program touches a page to fault it in: the region is wired once, with the region
   (data-model §2).
3. **Unrolling by modulo variable expansion** (M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo
   scheduling, MICRO 1994).
   - For each value, L is the smallest lag at which every read of it in execution t happens before its write in
     execution t + L. The unroll factor u is the largest L. Tensor-parallel decode gives u = 1; recurrent state updated
     in place, 2; 1F1B pipelines, up to the number of stages. A value with no such L fails the plan with `ENOMEM`,
     naming the value.
   - The template is unrolled into u copies, and execution t runs copy t mod u. `executions` is no longer an input of
     `mesh_plan`: a plan runs any number of executions.
   - Interference is happens-before over the template's copies, tested at the distances at which their executions
     meet. A landing's write is set by another node's send, so lifetimes under happens-before are not intervals on one
     order, and left-edge doesn't apply. The colouring is greedy by size (Y. Pisarchyk and J. Lee, arXiv 2001.03288,
     2020), as `mesh_graph_left_edge` does now.
   - Its cost is compile time only, once per plan: quadratic at worst in one execution's values × u, never in the
     run's length. At run time there are no allocations and no bookkeeping; every address is a constant in the plan.
4. **Nothing grabs.**
   - A program holds exactly the pages its plan uses (`mesh_graph_left_edge`'s `used`). It claims them when
     `mesh_plan` places them and releases them when the plan retires. Nothing is claimed at attach, and any number of
     programs hold disjoint sections of a node's region at once.
   - A new plan's state is in new pages, never the previous plan's, so nothing erases a plan or waits for one to be
     erased.
   - The bridge's own turn releases a retired plan's pages once no port holds them. This is deferred reclamation (RCU's
     `call_rcu`, P. E. McKenney and J. D. Slingwine, Read-copy update, 1998; K. Fraser, Practical lock-freedom,
     2004), and nobody waits for it.
5. **The normal program demonstrates the reuse.** `e2b_replay` passes its program's data edges to `mesh_plan` in place
   of `NULL, 0` (e2b_replay.m:578–579). They come from the captured graph's stages: each stage's reads and writes, in
   the order its command buffer runs them. `tools/prepare_e2b.py` writes them into the manifest where it doesn't
   already. The run's pages then don't grow with its steps.
6. **Every race is closed and every wait is bounded.**
   - The cancel that lands after a re-plan zeroes the landed records: gone, because no plan writes another's records
     (items 4 and 7).
   - A port that keeps a retired client's plan past `MESH_PLATFORM_NS`: gone with the claim at attach (item 4).
   - The local countdown wait that ends with zeros and no mark: gone with the lane waits (item 1).
   - A link loss and a GPU bound that both read as `ETIMEDOUT`: the GPU bound is gone (item 1), and a link loss
     arrives on the count as cancellations.
   - The host's final completion wait in `e2b_replay.m`: bounded.
7. **Nothing is zeroed or reset.**
   - **No data page is written** but by its producer or its landing.
   - **The landed list's phases** (PTX mbarrier's phase bit, widened to two):
     - Each receive in the plan's list (a received page of one value's copy) has a 2-bit phase, Gray-coded
       00 → 01 → 11 → 10. A landing is one atomic XOR of one bit, which never carries into the next receive's
       field. Its one writer is the bridge's data path that takes the receive.
     - A reader of execution t compares the field with Gray(⌊t / u⌋ + 1 mod 4): equal, landed; one behind, not yet;
       anything else, a write ran ahead of the allocator's order, which is an error.
     - The fields start at 00 when the allocator places the list, once per plan, as `mbarrier.init` runs once per
       kernel. A field is reused only with its page, so the allocator's order for the page covers it.
   - **A supplied word** has one writer, its range's producer, which stores t + 1 after its last write in execution t.
     A send of execution t goes once its word reads at least t + 1. The word starts at 0 when placed, and one word
     serves every execution.
   - **The landed count and the queue's event** only rise.
   - **The only zeroing left** is mesh-link.c's zeroing of the region's header, once, when the bridge creates the
     region.
   - **The projection kernels' arrival counters** already have this form: the last arrival restores the count
     (native_projection.metal:232), as an mbarrier's phase completion reloads it. Their partial sums need no zeroing,
     since every threadgroup writes its own before any are read.
8. **What judges it.**
   - One configuration runs in a runtime, and no call site handles two forms.
   - The builds are clean: `make -C rdma` and `make -C rdma native-audit` in mesh, and `make .build/e2b_replay` in
     metal-microbench.
   - Parity and the reuse are shown by the normal program, when the operator authorizes runs: e2b_replay's results,
     and pages that don't grow with its steps.
   - No data page is written but by its producer or its landing. No state word is reset while its plan stands, and
     each has one writer; counts take arrivals.
   - No lane waits. Every wait that remains is bounded, and every failure is an error value.
   - Commits are pushed to main as each piece builds.

### C4. What goes

**mesh**
- rdma/mesh-collective.c: `executions` as an input of `mesh_plan`, and its per-execution items; the `memset` of every
  declared instance's pages before `hops` (lines 749–757); the word-by-word zeroing of the supplied words and landed
  records (lines 759–763).
- rdma/mesh-call.h and rdma/mesh-dataflow.c: `mesh_plan_erased`, and its seq_cst fence pair with the ports' data path
  (rdma/mesh-flow.c).
- rdma/mesh.h: the supplied words' and landed list's fixed areas of the layout (hdr `word_off`, `landed_off`); the
  landed record's ORed bits and its `UINT64_MAX` cancel word; the constant `MESH_SUPPLIED`.
- rdma/mesh-dataflow.c: `mesh_claim`'s claim at attach of everything free, and the wait before it for the ports to
  drop a retired client's plan; the claim's local run, where values were laid out when created
  (`mesh_section_create` with `wire` 0).

**metal-microbench**
- metal_recording.m: both lane waits in `mesh_read_N` bounded at 2^23 polls (lines 1921–1996), the remote and the
  local countdown; `mesh_seen`'s failed value; a lane's `UINT64_MAX` into a step's landed record, and the host's read
  of it; the zeroing of the projection kernels' partial sums (line 1546).
- tools/e2b_replay.m: `NULL, 0` at line 579; the refusals where steps differ from executions (24ec8cda, cfada099, and
  472390a8's comment); the zeroing of every local operand's whole extent (line 627) and of every pooled intermediate
  (line 840).

## Appendix D: the operator's rulings on the revision

The revision's implementation (mesh 14fcfcd7, metal-microbench a062db81; design/disjoint-pages-revision-coordination.md)
and its review on 2026-11-04: Claude's report and the operator's turn, verbatim (D.1); Claude's answers to the
operator's two questions (D.2); Claude's proposal (D.3), the unit of work; and what goes (D.4). The departures the
report numbers 1 to 4 are departures D8, D11, D12 and D7 of design/disjoint-pages-revision-coordination.md. Where D.3
differs from C3 or from anything earlier in this document, D.3 holds.

### D.1 The turns

**Claude** (the implementation report):

> The revision is implemented and pushed in both repos, and all three builds are clean. Nothing has been run, so parity
> and the page reuse are untested until you authorize runs.
>
> The coordinator used 17 subagents: 14 implementers, each owning its own files, and 3 reviewers. Its contract and the
> full routing record are in `design/disjoint-pages-revision-coordination.md`, which is listed in INDEX.md. Both repos
> are level with origin: mesh `14fcfcd7`, metal-microbench `a062db81`.
>
> **What's done.** Every C4 entry is gone. I spot-checked that the lane poll loops and the replay's memsets no longer
> exist. The `memset`s left in `mesh_plan` cover only the plan's state section and publication records, once per plan.
> No data page is zeroed.
>
> **Departures that conflict with your rulings:**
> 1. **One client at a time.** A second program to attach gets `EADDRINUSE` (mesh-dataflow.c:215). That contradicts
>    C3.4 ("any number of programs at once") and your rule against grabbing. The coordinator says serving several
>    clients needs one page list per client, which C3 doesn't name.
> 2. **LiteRT's weights, and some delegate buffers, stay outside the allocator.** That contradicts C3.2. The
>    coordinator's reason is that placing them would mean writing their contents, which C3.7 forbids. That misreads
>    C3.7: loading the weights is the weights' producer writing them, which C3.7 allows.
> 3. **Per-step host inputs still grow with steps.** That covers positions, masks, parameters and the schedule, and
>    contradicts C3.5. The coordinator says C3 doesn't say how a host producer is ordered against the GPU. C3.1 does: "A
>    host producer is ordered by the event."
> 4. **The Metal 4 submission was dropped for one Metal 3 command buffer.** The reason given is that
>    `encodeWaitForEvent:value:` exists only on `MTLCommandBuffer`. But `MTL4CommandQueue` has `waitForEvent:value:`
>    (MTL4CommandQueue.h:274). It waits between commits rather than inside a buffer, so the Metal 4 path could have
>    stayed with one commit per wait. Whether this costs speed is unmeasured.
>
> **Interpretations that look sound to me:**
> - Landed counts are kept per received range rather than per node, because e2b's dependent exchanges would otherwise
>   hold every pass until its deadline.
> - Each run's execution count goes to the transfer start instead of the plan.
> - The first pass masks whole threadgroups, because the kernels synchronise within one.
> - The supplied-word add is 32-bit, because Metal Shading Language has no 64-bit add.
> - A node that has failed an execution stops releasing its sends.
>
> **Still open:**
> - **A system call on every landing.** Each completed range makes one wake call, and for e2b's one-page partials
>   that's every landing. Its cost is unmeasured.
> - **Unproven driver behaviour.** The bridge re-posts a receive record while an earlier post of it is outstanding.
>   That's safe only if the provider copies the request when it's posted, which the disassembly suggests but doesn't
>   prove.
> - **Unrun kernels.** The spliced kernels are compiled only at capture, so some exist only as read code.
> - **Wrong error after a relay's loss.** Nodes downstream of a relay report a timeout instead of a lost link.
> - **Older graph walks.** The C9 hook flagged walks in metal_recording.m that predate this change.
> - **Stale doc.** `design/prepared-machine.md#M08` still describes the old receive record.
>
> The scratchpad held 167 stray files from subagents, 84 of them copies of source files. I deleted them; the source is
> in commit history.
>
> Items 1–4 are the next unit of work. I can write them up as the next appendix to the change request when you want.

**The operator:**

> "- Unproven driver behaviour. The bridge re-posts a receive record while an earlier post of it is outstanding. That's
> safe only if the provider copies the request when it's posted, which the disassembly suggests but doesn't prove." is
> this beahvior required or a fancy of someone who really wanted to have a complicated base case
> "- Wrong error after a relay's loss. Nodes downstream of a relay report a timeout instead of a lost link."
> similar issue. otherwise address 1->4. no metal 3 commandbuffers.

### D.2 Claude's answers

**The receive posted again while an earlier posting is outstanding is not required.**
- The hardware holds a SEND in credit until its receive is posted (data-model §3). A SEND that comes first waits; it
  isn't lost. Posting ahead saves that wait and nothing else.
- The allocator orders the peer's SEND of execution t + u after this node's reads of t, and those reads come after t's
  landing (C3.3). So a record posted again as soon as its own last posting lands is always in time.
- The memory a posted receive names is the plan's page itself: its one scatter element is the page's address in the
  registered region (mesh-flow.c link_prepare, `wire_span` of `hop->page`; mesh-verbs.h:56). Nothing is staged or
  copied. The hardware may write that page until the receive completes, so the page is named in a new receive only
  after the last one completes. The verbs interface calls that memory the request's "data buffers" (the SDK's
  infiniband/verbs.h, at ibv_post_send).
- The early second posting bought nothing. It put two postings on one page, and it rested on the provider copying each
  request when it is posted, which nothing documents.

**The timeout after a relay's loss is not required either.**
- It comes from treating a loss as something only the cable's two ends learn. They cancel the receives from the lost
  peer, and nobody else cancels anything. A cancellation writes no phase, so the relay never releases its forwards, and
  the node past it waits out its deadline (MESH_PLATFORM_NS) and reads ETIMEDOUT.
- But every loss already reaches every node. The relay's record loses the cable, the map serial moves, and the record
  floods to every bridge (RFC 2328 §13; mesh-records.c map_put). A plan is made against one serial, and a port already
  refuses to load a plan whose serial is stale (mesh-flow.c link_prepare, ESTALE).
- So one rule at every node gives every node ENOLINK within one flooding, with no relay case: a plan stands only while
  its map does. It is link-state routing's own rule: routes come from one version of the database, and a change
  recomputes them (RFC 2328 §16).

### D.3 Claude's proposal

Items 1 to 4 are the report's departures, in its order. Items 5 and 6 are the answers above. C3's four rules hold
throughout.

1. **Any number of programs at once.** (Departure D8.)
   - **Why it needs a name.** One queue pair carries one line of receives, filled in the order they were posted, and
     there is no shared receive queue (data-model §3). A Thunderbolt device has 11 queue pairs, 10 of them usable
     (RDMA-FIRST.md). So two programs can share a link only on disjoint queue pairs, and the link's two ends must agree
     on which program holds which. Only a name both ends know can tell them. NCCL names a communicator by the
     `ncclUniqueId` the application gives every rank (`ncclCommInitRank`); PMIx names a job by its namespace.
   - **The name.** `mesh_plan` takes the program's group, a 64-bit value that is the same at every rank, picked once by
     the program's launcher. A plan is named by (group, epoch). The epoch is the client's count of its own `mesh_plan`
     calls, which is the same at every rank because the call is collective. The launchers (tools/mesh/run_pair.py and
     tools/mesh/grid.py) pass one group per launch to every rank, as `MESH_GROUP`.
   - **Attach takes nothing.** There is no client word, no compare-and-swap and no EADDRINUSE. A client is named only
     in the heads of its claims (its pid and start), and only once it plans.
   - **Each plan in its own claim.** A plan's entries and its state section are placed by its allocator in its own
     pages, as the state section is now. The region's header holds no plan: `hops`, `plan`, `state` and `client` go.
     One word, `plans`, rises and is woken whenever any plan is written or retired. The bridge then finds the standing
     plans from the claims' heads.
   - **Lanes.** On each link it crosses, a plan holds the lanes its units use, apart from every other plan's at both
     ends. The link's lower-numbered node chooses them from the lanes free at both ends, up to the plan's units on that
     link in one copy, and names them to the other end over the control socket by the plan's name. Each end frees a
     plan's lanes when its own serving of the plan ends with nothing outstanding, and tells the far end. Where no lane
     is free, that port refuses the plan with EBUSY, and the client's start returns it. Queue pairs stay the bridge's;
     no program sees a lane.
   - **Serving.** A port serves every standing plan that holds lanes on it, each plan's records on its own lanes, with
     its own cursors and counts.
   - **Announcements.** A port's load of a plan sets the port's bit in the plan's own state (one atomic OR, a bit per
     port). The release of a retired claim reads them there. This is the coordination's D10, moved into the plan's
     state.
   - **Holders and retirement.** The heartbeat's holders are the clients the live claims name. A gone client's claims
     are retired on the bridge's turn, as now.
2. **Every value placed, the delegate's included.** (Departure D11.)
   - Every buffer the capture binds is a value of the plan: the weights, and every delegate buffer the recorder doesn't
     rename, as well as those it does. A buffer written before the first execution and read by every one is an
     invariant (C3.2). That covers the weights, the constants, and any buffer whose first read in an execution comes
     before its first write. The rest are execution values at their levels.
   - An invariant's producer writes it into its placed pages once, after the plan and before the first execution. For
     the weights, that write is the load: one copy from the buffer the delegate loaded, at the bind. C3.7 allows it,
     because the producer is writing its own value.
   - After the bind, no command reads a delegate buffer.
3. **Host inputs in copies, ordered by events.** (Departure D12.)
   - The per-step host inputs (positions, masks, cache parameters, the schedule's per-step values, the token stream)
     are execution values with copies, not one instance per step. The program declares the host's lag, and the
     allocator gives them copies by it, as modulo variable expansion does for any value. With the host writing step
     t + 1 while the GPU runs step t, the lag is 2.
   - The host writes step t's inputs into copy t mod k (k being their copies) once the queue has signalled that step
     t − k is done, then sets its input event to t + 1. Step t's commands that read them are committed after the queue's
     `waitForEvent:value:` on that event at t + 1. The queue signals each step's end with `signalEvent:value:` after its
     last commit. The host's wait on that is bounded by MESH_PLATFORM_NS and ends as an error value.
   - Each event has one writer: the host sets the input event, the queue sets the step event, and the client's waiter
     sets the landed event. This is C3.1's "A host producer is ordered by the event".
   - Nothing the run binds grows with its steps.
4. **Metal 4 submission. No Metal 3 command buffers.** (Departure D7.)
   - Every command buffer this code makes is an MTL4CommandBuffer from an MTL4CommandAllocator, committed to an
     MTL4CommandQueue. That covers the decode, the prefill, the preparation and fills, the profiling, the recorder's own
     copies and tools/native_bench.m.
   - The queue holds each later pass. The commands before it are committed. Then comes `waitForEvent:value:` on the
     landed event at p + 1 (MTL4CommandQueue.h:274). Then the later pass and what follows go in the next command
     buffer. That is one commit per later pass, and the wait is the GPU queue's, not a host thread's.
   - A completion is the queue's `signalEvent:value:` (MTL4CommandQueue.h:266) and a bounded wait on that event, or an
     MTL4CommitFeedback.
   - One Metal 3 interface stays, and it is the delegate's: ml_drift encodes into an MTLCommandBuffer it is given, and
     commits its own loads during the capture. The recorder's queue for the delegate exists only for the capture.
     Nothing the run executes passes through it.
5. **A receive posted once per landing.** A lane posts a record's round j + 1 only once that record's `landings` reads
   j + 1. That is one compare, in the line the post already reads. A queue pair's receives complete in the order
   posted, so when a record lands, every record before its next round on the lane has landed too. The lane never waits
   on anything but the record itself.
6. **A plan stands only while its map does.** Where a node's map serial moves away from a standing plan's serial, its
   bridge cancels every receive of that plan that hasn't landed (plan_cancel over every entry: each an arrival on its
   count, no phase written). Every node of the plan then reads ENOLINK. The cancellation of the lost peer's receives at
   the cable's two ends stays.
7. **What judges it.** C3.8, and also:
   - A second program attaches and plans while the first runs, on its own pages and lanes. The first's end leaves the
     second running.
   - This code creates and commits no Metal 3 command buffer but the delegate's capture.
   - Nothing bound grows with the steps: not the routed values, not the host inputs, not the delegate's buffers.

### D.4 What goes

**mesh**
- rdma/mesh.h: hdr `client`, `hops`, `plan` and `state` as the one plan's; the layout's fixed page-list area
  (`hop_capacity`), whose entries move into each plan's claim.
- rdma/mesh-dataflow.c: the client word's compare-and-swap in `mesh_attach`, and its EADDRINUSE.
- rdma/mesh-flow.c: a receive record posted while its earlier posting is outstanding, and link_prepare's note that
  relies on the provider copying a request; the one flow per port.
- The ports' `prepared` in the link table: the announcements move into the plan's state.
- design/disjoint-pages-revision-coordination.md, "Open at the end": the item on the provider's copy of a request.

**metal-microbench**
- Every MTLCommandQueue and MTLCommandBuffer this code creates outside the delegate's capture: tools/e2b_replay.m
  (`commit_bounded`, and lines 1048, 1069, 1118 and 1277); metal_recording.m (`MetalCompletion` and `MetalCompleted`
  outside the capture, line 492, and the `id<MTLCommandBuffer>` of `MetalReplayRecorded` and `MetalReplayEncode`);
  tools/native_bench.m (lines 97 and 466).
- The per-step instances of the host inputs, and their bindings at t × stride.
- The delegate's buffers bound as captured, and `MetalNativeSubstitute`'s reads of the weights from them.
