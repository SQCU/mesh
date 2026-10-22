# mesh-lint: the rules a review found, kept as checks

`bin/mesh-lint` reads the bridge, the collective and their clients, and reports each deductively determinable
violation as `path:line: RULE: message [source]`. `make -C rdma lint` runs it.

Why it exists, in the operator's words (2026-10-22): "i'm tired of this runaround where one agent writes extremely
incorrect collective code then another agent 'notices' what was extremely incorrect, 'fixes' the code, then stops
existing and leaves no guidance or corrective or example in the repo". A review that finds a new class of defect adds
its rule here and to `bin/mesh-lint` in the same commit as the fix.

Rules a pattern decides run every time. Rules marked *semantic* need a function to be read: `bin/mesh-lint --semantic`
asks a model about one function at a time. It is given this file's rule cards and the vocabulary below, and caches its
answers in `rdma/.build/mesh-lint-cache.json`. The operator: "a parser can use language model calls on compact units of
text or source if it reduces implementation complexity and delivery speed by an order of magnitude".

Performance is the requirement. The operator: "performance is harder than safety and produces safety anyways most of the
time, and we're requiring performance rather than safety." Every rule below costs time, removes a wait, or keeps
backend complexity out of linear algebra.

## Rules

### Time and waits
- **T1-CLOCK.** A clock reading, or a `*_NS` bound, decides a branch in a data-path function (`DATA_PATH` in
  `bin/mesh-lint`).
  - Source: the operator, 2026-10-22: "we don't introduce race conditions or use timing anywhere ... using timing is
    always slower than not using timing".
  - Wrong: `if (now() > deadline) send_dud();`
  - Right: wait on the event. That is a count reaching its target, a stamp, or a command buffer's status. A finite
    slice only re-checks a revoke (design/rubrics.md M3).
  - A measurement that decides nothing (a time ring) carries `meshlint: measure`.
- **T2-SLEEP.** A sleep, or a timed wait, in a data-path function. Same source.
- **K1-SPIN.** A spin inside a kernel (`for (...;;...)` or `while` around an atomic or coherent load) other than the
  library's landing wait (`mesh_wait_bounded`, `mesh_step_await` and their kind).
  - Source: the operator, 2026-10-21: "keep only the landing wait, one acquire fence, a poll bound for the node".
  - Wrong: a grid barrier on stamps inside a step kernel.
  - Right: data a later stage of the same dispatch needs is made where that stage runs, or the stage is a dispatch of
    its own.
- **P1-SYNC.** A host sync in the step loop: `synchronize`, `.item()`, `.cpu()`, `.tolist()`, a barrier, or a blocking
  `copy_`. Functions the loop calls are included.
  - Source: the operator, 2026-10-22: "algorithmically unjustified halts, waits".
  - A ring's flow control, which waits for the step that last read a slot, carries `meshlint: ring`.
- **P5-CLOCK.** A clock read in the step loop. Same source as T1.

### Errors and guards
- **M3-QP.** A queue pair moved to RESET or ERR, or destroyed, inside a running process.
  - Source: design/rubrics.md M3. The operator, 2026-10-22: the documented flush is the only error-state use, and it
    is unused because the plan's window bounds what is in flight (design/extents-and-liveness.md, d).
- **K2-ERR.** An error word or a patience bound inside a data kernel.
  - Source: the operator, 2026-10-22: "error states can be made unreachable and proven to be unreachable".
  - Wrong: `if (polls > PATIENCE) { stop = 1; err |= cause; }`
  - Right: no such state. Waits are the library's landing wait, whose poll bound is the node's.
- **semantic SEM-META.** Error metadata consumed inside a call path, to branch, refuse or wait.
  - Source: design/pages-and-functions.md "Asynchronous error metadata".
- **semantic SEM-GUARD.** A check or refusal that guards a state the plan excludes, with no rubric citation.
  - Source: AGENTS.md "Rejection filtering and defensive programming are forbidden".

### Coherence and memory
- **K3-STORE.** A plain store to device memory, such as `simdgroup_store`, in a kernel that publishes.
  - Source: metal-microbench docs/measurement.md 2026-09-29: the NIC reads a plain store only once its command buffer
    has ended.
  - Right: `mesh_step_write`.
- **K4-LAND.** Landed bytes read without `mesh_step_read`. Source: nccl-mesh-kernels.h STEPS.
- **semantic SEM-XTG.** Data written by one threadgroup and read by another within one dispatch, through memory that
  is not `coherent(device)`.
  - Source: MSL; mesh ebdef267. It held on one die and failed on the M5 Ultra's two.
- **L1-LINE.** Two words with different writers share a cache line (`hw.cachelinesize`; 128 bytes on Apple silicon).
  The writers are the table `WRITERS` in `bin/mesh-lint`; the offsets come from a compiled `offsetof` probe.
  - Source: bridge audit 2026-10-22, item 3.
- **S1-SEQCST.** A sequentially consistent atomic on the per-unit path: a store then a load, so the core waits for the
  store to be seen. Source: bridge audit, item 4.

### Units, placement and planning
- **U1-PAGE.** A unit's length is the whole page, not the plan's bytes.
  - The operator, 2026-10-22: units are page-start-aligned and express only a length. The length is typed, and checked
    when the plan is compiled against the page length of the receiving peer. Page length is never assumed uniform
    across the mesh. Only collective code knows any of this.
- **D1-PAGE.** Data code reads a collective backend's page facts (page size, page rounding).
  - Source: the operator, 2026-10-22: "data processing code may never discover these abstracted features";
    "information hiding from data processing code authors on some level must exist for performance reasons and for
    decoupling backend complexity from linear algebraic function complexity rather than for 'security' or 'memory
    safety'".
  - Right: the collective's allocation gives the buffer.
- **P2-LAND1.** One landing buffer reused every crossing. The landing is then gated on this rank's own release, so each
  step waits for a receive post and a credit. Source: bridge and GPU audits 2026-10-22, item 2. Right: a ring of two
  or more slots, each gated on its last reader.
- **P3-REPLAN.** A crossing planned again every step.
  - Source: host audit 2026-10-22, A.1; design/extents-and-liveness.md, a.
  - Right: presented once and started each step, as `MPI_Allreduce_init` and `MPI_Start`.
- **semantic SEM-COMPLETE.** A call that completes unrelated earlier groups before its own work, so a block-ahead
  prefetch is waited out on the critical path. Source: GPU audit 2026-10-22, item 3.
- **semantic SEM-WAIT.** A wait on anything but data: a message about data, a flag, a timer.
  - Source: design/pages-and-functions.md, the addendum on messages about data.

### Terms
- **semantic SEM-TERM.** A coined term, a metaphor, where the literature has a term. The model names the literature's
  term.
  - Source: the operator, 2026-10-22: "neologisms always appear when agents are confused and wrong about an api ...
    every new neologism i'vce heard in the last week has been related to a regression in performance". See Derrida,
    "White Mythology: Metaphor in the Text of Philosophy" (1971): a worn metaphor passes for a concept while hiding its
    origin.

## Vocabulary
Terms the literature uses for the mechanisms here. A semantic finding proposes one of these in place of a coined term.
- **Verbs (IBA, rdma-core, TN3205):** work request, post (post_send, post_recv), SEND, receive, completion, completion
  queue, poll, queue pair, send queue, receive queue, memory region, scatter/gather entry, frame, credit, flush.
- **Metal:** command queue, command buffer, commit, encoder, dispatch, threadgroup, simdgroup, residency set, shared
  event, hazard tracking, `coherent(device)`, `coherent(system)`, fence, barrier.
- **NCCL and MPI:** communicator, rank, group, send, recv, all-reduce, all-gather, reduce-scatter, broadcast, persistent
  request, start, partition, ready, revoke.
- **The memory model (design/pages-and-functions.md):** page, row, stamp, use count, page table, release, extent,
  function, NFE.
- **C11 and C++:** atomic, load, store, acquire, release, seq_cst, fence.

## Adding a rule
When a review finds a defect class:
1. write its rule card here: id, what, source, wrong and right;
2. add its check to `bin/mesh-lint` (`check_c`, `check_metal`, `check_python`, `check_lines`), or mark it *semantic*;
3. fix the code;
4. commit all three together.
