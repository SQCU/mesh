# Integration readiness, 2026-10-29

A readiness review of the other group's integration work over one calendar day: Codex thread 01a16b5d, 94 commits
on main between 2026-10-28 09:24 and 2026-10-29 10:03 (local), 37 in metal-microbench (mb) and 57 in mesh. Each mb
commit is paired with a mesh commit that records the same step in design/operand-contract.md. The operator's second
Codex thread (01a100d2) works in ~/vllm and committed nothing here. Commits were attributed from the thread's own
commit output, not by message style.

## The assignment

The thread's goal (set 2026-10-27 12:44): finish the lowering of "compressed cuts, sparse tails, specialized ANE
constraints, and arbitrary page mappings", which is the current pass of step 9 of its nine-step plan ("general
backend/layout integration and constraint propagation"), then integration gaps 2, 3 and 6. Step 2 is the collective
boundary that joins all preceding host work (metal-recording.m). Step 6 is LiteRT's packaged delegate, which leaves
out internal tensor views within shared arenas. The plan after those is a static integration review, then preparation
of the complete graphs across the four-node configurations, then canonical whole-function runs. Until then the thread
verifies by "source review, builds and graph preparation" only. That follows the operator's direction of 2026-10-28:
no previous code must run until the one-sided-write, unified-use-of-pages data model has been imposed on the source.

## What the work is, in plain terms

One idea runs through all of it. Every operand (a weight, an activation, a collective's send or receive buffer) is a
declared view over region pages: a relation from logical element to page and offset. Preparation compiles that
relation into index arithmetic or a compact constant table that the kernel or WRITE descriptor reads directly. Each
commit takes one place where code still rebuilt a layout, staged a copy, loaded an order buffer, capped a width or
fell back to a dense buffer, and routes it through the declared relation instead.

- **The index algebra (the page indexing):** rdma/mesh-tensor-layout.m, +1297 −215. Cuts, permutations, reshapes,
  and reversed, repeated or compact-stripe selections compose as prepared index arithmetic without expanded tables.
- **The collective side (the route compiler, mapping and coalescing):** mesh 10-29 02:50–04:48.
  - WRITE descriptors take their offsets from the bridge page table, so scattered pages need no contiguous lease
    (5daa61c9).
  - Backing and registration cuts are joined across all declared slots and relays, so no packet crosses a backing
    run or an extent (0e298327, 4b27988f).
  - Numerical readiness is compiled from packet lengths: element j waits on the landed prefix of the packet holding
    its last byte (rdma/mesh-landing.h, 29c9d398). Every reduction combine reads those packet-ready operands
    (a2bdd487).
  - Scattered bridge pages are mapped once into one local virtual interval, which CPU, Metal and CoreML share without
    a copy (f77cf2c1).
- **The native engine and LiteRT side (mb):** the e2b graph's provider storage becomes declared views.
  - Compressed weights take arbitrary column selections.
  - ANE shares are compiled independently of cached block boundaries.
  - Attention and partial-merge widths lose their caps; the model head-width filter is deleted.
  - The forced plain-Metal buffer and the implicit dense-buffer path are deleted, so a missing relation now fails
    preparation (e807ac1d).

## Readiness

- **Builds:** the thread reports a build of the native libraries, rank and Goal A at each commit. The hourly checkouts
  built mesh main through the window.
- **Run-time evidence for the mb half:** none, by direction. No hourly workload (bandwidth, chain, lstm, cp, tp, ep,
  fabric-dnn) runs the e2b or LiteRT path, and the thread has run no numerical execution. Its only evidence is
  pass/fail preparation checks.
- **Run-time evidence for the mesh half:** incidental, through the hourlies, which run torch_mesh and libnccl-mesh.
  Passes went from 7 of 17 cases (10-28 16:28) to 12–14 (23:31–05:46), then 6 at 09:50 with one node outside the
  linked fabric.
  - The recurring failures are links lost and paired again, GPU waits reaching their bound, placement unmet, and
    guard kills.
  - GPU-wait and link-loss failures hold at 2–3 cases per run from 10-28 16:28 onward, before the collective commits
    above, and those commits do not change the count. None of the failures bisects to a commit of this thread in the
    ledger.
  - One Metal compile error ("cannot assign resource locations to 'words'", chain-n2 cases) appears in the 04:36 run
    only and is gone at 04:46.
- **Convergence:** each commit closes one case and records the cases it found next. design/operand-contract.md went
  from 383 to 1726 lines over the day. Its lines that say what remains ("remain", "still need", "not yet",
  "unfinished", "incomplete") went from 15 to 67. The goal's four named terms each have commits, but the goal stays
  open because "general" and "arbitrary" name no closing set.
  - Now listed as remaining: general native declarations of logical layout and scalar encoding, input and output
    placement for collectives that change shape, external and signature selections, and selections across resized
    source domains (operand-contract.md, "Partial contraction boundaries").
  - Steps 2 and 6 are still queued. Since 07:16, LiteRT internal views (step 6's subject) have been worked under
    step 9.
  - The distance to the first run is therefore not bounded by anything the thread has stated.
- **Size:** code +8844 −4951, net +3893 (mb +1516, mesh +2377), and docs net +1809. The deletions are real
  (native.metal −405; the order and identity buffers; the dense fallback; the width filter). The growth is mostly the
  index algebra and the LiteRT preparation tools (tools/prepare_e2b.py +810 −377, tools/litert_storage.py +299).
- **The file split of 2026-10-28:**
  - The collective core (nccl-mesh.c, mesh-span-plan.h, mesh-routed.c, beneath nccl.h's signatures) went to the
    Claude group. This thread edited nccl-mesh.c in 6 commits, nccl.h in 3, mesh-span-plan.h in 4 and mesh-routed.c
    in 1.
  - Both groups edited nccl-mesh.c in the 02:30–05:10 window. At least four of the thread's mesh commits landed
    under hashes other than the ones it first created (5daa61c9, 0e298327, 4b27988f, f77cf2c1).
  - nccl.h's ncclMeshCommPlan `bindings` changed from an optional `const ncclMeshBuffer*` ("NULL: none, as before")
    to `const ncclMeshBufferSlots*`, which execution now requires. That agrees with the no-eager mandate (bindings
    declared first), and it changes the C–D interface.

## Rankings (design/rubric-actionable.md)

| Kind | Dimension | Pair | Deciding code |
|---|---|---|---|
| D | D2 when choices are made | old ≺ new | relations bound at preparation; a missing relation fails preparation, not the call (e807ac1d, 85ae793f) |
| D | D4 one implementation | old ≺ new | `ScaledReduction` replaces `OrderedReduction` and its identity buffer; the plain-Metal and dense paths deleted |
| C | C3 routing as data | old ≺ new | readiness runs and backing cuts compiled from page tables and packet lengths (mesh-landing.h, 0e298327) |
| M | M2 messages about messages | unchanged | readiness is local arithmetic over landed prefixes; nothing new crosses a link |
| D | D9 evidence | unchanged (none for the native path) | preparation checks are pass/fail; deferred by the operator's direction |

The first three rank the new form above the old in each kind. The open question is convergence, not direction.
Integration that records more remaining cases than it closes, judged only by preparation checks, cannot bound how far
it is from the first run. A closing set for step 9 (the cases the four-node graphs actually prepare) would bound it.
