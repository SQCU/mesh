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
- **Run-time evidence for the mb half:** none. No hourly workload (bandwidth, chain, lstm, cp, tp, ep, fabric-dnn)
  runs the e2b or LiteRT path, and the thread has run no numerical execution. Its only evidence is pass/fail
  preparation checks.
  - The absence comes from the hourly runner, not from this thread. The runner's case list has been narrowed and
    reverted repeatedly: cp and ep limited to two nodes and restored (889e3cb2, b0262245, 08726d3d, 695315cc,
    1ed7a520, 9040b674); duplicate cases deleted again today (df2a51d0).
  - tools/torch_parallel/e2b.py was deleted on 10-22 (c3924775).
- **Run-time evidence for the mesh half:** through the hourlies, which run torch_mesh and libnccl-mesh. A case's
  evidence is its (µs, GB/s) tuple; how many cases completed is not evidence. Here are the medians across the window's
  runs, before and after the thread's first collective commit (10-29 02:50), with the before spread:

  | Case | GB/s before (min / median / max) | GB/s after (median) |
  |---|---|---|
  | chain-n2 | 0.385 / 0.388 / 0.591 | 0.389 |
  | chain-n2-w128 | 0.072 / 0.078 / 0.101 | 0.077 |
  | chain-n3 | 0.279 / 0.337 / 0.495 | 0.330 |
  | chain-n3-r32 | 0.042 / 0.740 / 1.20 | 0.792 |
  | ep | 1.52 / 2.69 / 3.13 | 2.59 |
  | tp | 0.167 / 0.187 / 0.200 | 0.191 |
  | lstm-n2-h128 | 0.0102 / 0.0126 / 0.0163 | 0.0133 |
  | lstm-n2-h16384 | 0.084 / 0.099 / 0.141 | 0.104 |
  | lstm-n3 | 0.033 / 0.038 / 0.042 | 0.039 |

  - Every after-median lies within its case's before spread, so the commits moved no rate outside the noise.
  - The link and GPU-wait faults the cases end on recur at the same rate before and after the commits. None bisects
    to this thread.
- **Convergence:** design/operand-contract.md went from 383 to 1726 lines over the day. Its lines that say what
  remains went from 15 to 67. That growth alone does not show divergence.
  - The cases named as remaining are reversed and repeated selections, selections across resized source domains,
    shape-changing collective outputs, and external and signature selections. Each is one composition, inverse or
    image in a closed, decidable algebra (design/lowering-closure-2026-10-29.md).
  - The work converges once that algebra's inverse is general. MetalTensorLayout now has three partial inversions in
    its place.
  - The closing set is the operator's: typed MPS programs, every `torch.export`/FX program for MPS and ANE
    collectives, the whole LiteRT engine on Metal, and a non-transformer DNN demo. Each is a finite operator
    inventory over that algebra.
  - Steps 2 and 6 are still queued. Since 07:16, LiteRT internal views (step 6's subject) have been worked under
    step 9.
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

The first three rank the new form above the old in each kind, and the direction is right. What would speed it is
capturing the algebra once (the relation type, its closed operations, emission, sizing, pages and data-dependent
operands). The per-operator tables of the three frontends can then be ticketed in parallel
(design/lowering-closure-2026-10-29.md, "The work, as tickets"). Each ticket sits outside transport and ends at
CommPlan's operands.
