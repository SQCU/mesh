# Change request: compile a synchronization signature and observe its execution

## 1. The compiled signature

The compiler produces ordered synchronization operations, counts, edges and inter-execution delays from the retained
scheduled IR. Each operation carries its source dependence and emitted-site correspondence. The graph module supplies
Bhattacharyya, Sriram and Lee (1997)'s synchronization graph and §7's redundancy analysis; Lee and Messerschmitt
(1987) supply delay semantics and Lam (1988) supplies unrolled copies. The signature consumer uses those solved
relations through [composition item 6](compilation-composition-change-request.md).

The signature covers submissions, queue signals/waits, encoder fences/barriers, host waits and bridge coordination.
The artifact carries its SHA-256; the printers and external monitor consume the same compiled result.

## 2. Three representations

The signature printer produces Matrix Market coordinate format (Boisvert, Pozo and Remington, NIST 1996), with
`i j delay` for each edge and index annotations identifying processor, IR vertex and operation.

The graph printer produces DOT with command-queue, thread and bridge clusters. Graphviz uses Gansner, Koutsofios,
North and Vo (1993)'s layered layout; `tred` exposes transitive relationships by Aho, Garey and Ullman (1972).

The terminal printer produces Lamport (1978), Fig. 1's space-time diagram: processor columns, event vertices and
signal-to-wait edges, with rows in Kahn (1962)'s topological order and text graph rendering as in git `graph.c`.

## 3. External observation

A separate DTrace process observes Metal calls through the Objective-C provider and host primitives through the
pid provider (Cantrill, Shapiro and Leventhal, USENIX ATC 2004; Leucker and Schallhart 2009, outline monitoring).
The observed program executes its compiled entry; the observer consumes events at those API boundaries.

## 4. Event correspondence

The monitor specializes its actions from the signature (DTrace generated D programs; Jones, Gomard and Sestoft,
1993, specialization). It compares each observed synchronization with the expected operation in that processor's
order and reports a mismatch with its source stack. DTrace's `raise()` action terminates the observed program on
an unexpected synchronization (operator's approved monitoring specification, 8 November, this file at `cdd44406`).

At completion it compares counts and prints both signatures in item 2's formats. The review record receives the
artifact digest and observation. Source-dependence correspondence and the cited redundancy analysis expose changes
to the compiler's synchronization choices; observed correspondence exposes changes in execution.

## 5. Captured program execution

The existing execution path runs the captured program's compiled artifact with item 3's observer attached
(apitrace and GFXReconstruct capture/replay; operator's approved specification, 8 November). The observer compares
that execution with its compiled signature through item 4.

## 6. Kernel synchronization

The typed kernel lowerer accounts statically for each emitted barrier, fence and atomic operation in the same
signature, retaining its source correspondence through final emission (operator's approved specification,
8 November; [composition item 6](compilation-composition-change-request.md)).
