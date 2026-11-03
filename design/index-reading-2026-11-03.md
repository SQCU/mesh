# The documentation read from INDEX.md (2026-11-03)

A fresh agent's reading, at the operator's request, of the 22 documents INDEX.md lists and those they point to, with no
session notes, memory, facts.md, commit messages or code comments as sources. A record of what the documents say, not
a decision. Paths are mesh-relative with `design/` dropped; `mm/` is metal-microbench.

## The required collective path, as the documents state it
1. Every buffer a crossing touches is a lease in the node's pre-registered region (NCCL.md:436-439, 500-504).
2. The whole function is declared before it runs; undeclared eager calls are prohibited (eager-execution.md:10-23,
   164-170; NCCL.md:628-636).
3. Placement, routes and the call sequence are fixed before execution; nothing is decided where a call runs
   (whole-function.md:49-55, 297-300; NCCL.md:147-168).
4. Each piece is lowered to fixed, exact byte ranges on each physical cable; relays forward landed ranges directly
   (NCCL.md:158-162; whole-function.md:14-30).
5. Groups are encoded into the caller's Metal 4 encoder; publication is one system fence and one store of `ready`
   (NCCL.md:253-256, 272-276).
6. The bridge posts receives ahead into the receiver's own pages and each SEND once its producer made the data; no
   header, receipt or ACK (transport.md:165-176, 521-523).
7. Consumers wait only on `landed` counts; sums fold as contributions land (NCCL.md:295-301; zorvek.md:101-107).
8. A page is written again only after its last DMA read completed, shown by `sent` (pages-and-functions.md:106-120).
9. A lost link sets the counts to ~0; the group fails as a value; the caller makes the communicator again
   (NCCL.md:586-592; transport.md:647-650).
Invariants (deliverables.md:196-222): two-sided transfers only (I1); presence is the receive completion, no ACK,
credit or sentinel (I3); no reuse guard (I4); lifetime by reference count, pages not zeroed (I5); no numerics or
application code in mesh (I7, I8); the caller places work (I9); failure an out-of-band status, recovery between NFEs
(I12, I19); one metadata line read per event (I22).

## Contradictions between documents
- Who names cables: whole-function.md:5-30 and NCCL.md:123-135 (the collective binds each transfer to a cable) against
  zorvek.md:277-288, 330-335 (that is the defect; cables and lanes are the bridge's alone); heterogeneity.md:41 (one
  stream per node pair).
- How reuse is proven: pages-and-functions.md:106-113, whole-function.md:178-185, zorvek.md:108-118 (the bridge's
  `sent`; downstream proofs deleted) against data-model.md:120-122, distributed-reduce.md:65-105,
  eager-execution.md:73-79, extents-and-liveness.md:107-109 (downstream landings).
- Error state: transport.md:418, 569-574, 592 and extents-and-liveness.md:223-226 (no flush) against transport.md:609-611.
- Deadlines: NCCL.md:560-561, eager-execution.md:68-69, torch-mesh/README.md:50-53 (5 s) against transport.md:233-240
  (no clock bounds a wait) and NCCL.md:575-576 (until a landing or a revoke).
- Pools: NCCL.md:429-435 (a lease pool gated by ncclMeshMemReusable) against eager-execution.md:186-187 and
  hardware-and-discipline.md:73-74.
- Placement from observations: torch-mesh/README.md:230-231 (never) against heterogeneity.md:59, 67, 108-125 and
  whole-function.md:248-268; mm/docs/principles.md:112-113 against heterogeneity.md:22-23.
- Formats: a span 24 bytes (transport.md:41, 438-440) or 16 (NCCL.md:241-245; pages-and-functions.md:284-285); the
  region layout 154 or 162 (transport.md:7, 433; pages-and-functions.md:301); registration extents 1 GiB or 2 GiB
  (transport.md:19; data-model.md:59-60; etc/bridge/README.md:34-41, 126-129).
- Halt notices: extents-and-liveness.md:167-193 against transport.md:539-540, 553-555.
- Callers: torch-mesh/README.md:8-15, 90-110 against eager-execution.md:179-181; the xonotic solver in NCCL.md:622-623
  against deliverables.md:289; hardware-and-discipline.md:26 against HETEROGENEOUS-LINKS.md:232.

## Deleted mechanisms still described as current
The tracks pool (NCCL.md:319-372), `taken` waits (transport.md:235-236), mesh_stream_present, stream lists and pin
rings (transport.md:68, 346, 510), all-ones dud pages and zeroing (distributed-reduce.md:158-159, 229-230),
ViewWrites, FusedForm and torch_mesh.fused (partial-tensors-and-fx.md:116-121, 220-224; whole-function.md:116), one
SEND per 16 KiB page (quiet-link.md:79-80).

## References to nothing
transport.md sections 15 and 16 (INDEX.md:31-32, README.md:433); data-model.md section 9 (heterogeneity.md:41);
transport.md "What a landing writes" (distributed-reduce.md:94); rubric-actionable.md's D-dimension and later line
numbers 5 low (D1 is rubrics.md:443, not 438).

## Unfinished work the documents record
Transport: the Mini's kernel data aborts (transport.md:287-288), the HELLO's equal-block check (:485-487), a stream
live at one end only (:553-555). Links P1, P2, P4, P6, P7 (HETEROGENEOUS-LINKS.md:230-263) and peer-version logging
(:224). Library: unaligned buffers across 1 GiB extents (NCCL.md:163-166, 508-511), the not-implemented list
(:649-653), the partials merge unmeasured beyond two nodes (:192, 221-224). Removals: eager-path deletions
(eager-execution.md:171-183); zorvek.md's premultiplier and operand copies (:147-153), descriptor walk (:185-186),
2.2 s preparation (:322-323), fabric-dnn failing (:289-290); the solver to become collective-solver-2026-10-30.md's
packing (whole-function.md:243). Deliverables T6 replicas (deliverables.md:303). Liveness measurements
(extents-and-liveness.md:212-215). Per-tensor parts and over-decomposition (heterogeneity.md:133-139). Frontends
(partial-tensors-and-fx.md:84-88, 165-166, 193-194, 280, 287-289; torch-mesh/README.md:174-191, 241-244). Serving's
three boundaries (compiler-integration-2026-10-31.md:54-56; mm/docs/serving-page-operands.md:54-69). Older records R2,
N1/N2, X5, E1d/E8, F11 (algorithm-sources.md:160-161, 201, 484, 1344, 1426).
