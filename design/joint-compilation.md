# The compilation's free choices, and how its optimizers compose

Quotations are from these records, each copied by script from the source's lines:
- design/joint-optimization-sources-2026-11-07.md (J);
- design/completion-and-period-sources-2026-11-07.md (CP);
- design/process-network-sources-2026-11-07.md (PN);
- design/modes-and-strategies-sources-2026-11-07.md (MS);
- design/staging-sources-2026-11-07.md (ST).

Code is cited at mesh 551b8799 and metal-microbench e57e44a3. Where a method's citation appears only in the code's
own comment and not in a record, the text says "as the code cites it".

## 1. Three compiled schedules, one problem

The compilation produces three schedules. Each has a literature of its own.

1. **An execution schedule of the program's operations.** The schedule may fuse, compose or reorder operations.
   - Halide separates this schedule from what is computed: "decoupling the algorithm definition from its execution
     strategy" (J §6).
   - Unity's algebraic transformations are such rewrites: "A typical algebraic optimizer will fuse the MatMul and ReLU
     operators to remove redundant memory loads and stores" (J §1).
2. **A collective communication schedule.** It fixes each transfer's route, then its order and time.
   - TACCL solves "a bandwidth-relaxed version of the problem to decide on routing, followed by ordering heuristics and
     a second bandwidth-constrained problem to find a valid scheduling" (J §4).
3. **A sharding of operands and parameters over devices.**
   - GSPMD takes "a few annotations on how to distribute tensors, based on which GSPMD will parallelize the computation"
     (J §5).
   - Megatron's column and row splits are one such sharding (docs/principles.md §4).

The literature treats the three as one problem:
- Unity "jointly optimizes algebraic transformations and parallelization" over "a unified parallel computation graph
  (PCG), which simultaneously expresses the computation, parallelization, and communication" (J §1). Of solving them
  one after another, "optimizing independently (also referred to as 'sequentially')", it says it "can miss significant
  optimization opportunities", and "joint optimization is necessary to maximize performance" (J §1).
- CoCoNet has "a compiler to generate jointly optimized communication and computation GPU kernels", with "fusion or
  overlapping of communication and computation" among its transformations (J §3).

So the compilation's variables are the free choices of all three schedules together. Each cited author's objective is
one term or one constraint of that problem. None of them is the decision.

## 2. How subproblems compose

The literature names four compositions.

- **Joint.** One problem over the variables of several subproblems: Unity's "hierarchical search algorithm to jointly
  optimize algebraic transformations and parallelization" (J §1).
- **Nested.** "Bilevel optimization is defined as a mathematical program, where an optimization problem contains
  another optimization problem as a constraint" (J §7). Alpa is an example: its inter-op pass "invokes the intra-op
  compilation pass on each stage-mesh pair, to query the execution cost of this assignment", and the intra-op pass
  "reports the cost back to the inter-op pass" (J §2). Alpa states the price of the decomposition: "While the joint
  execution plan is not guaranteed globally optimal, they demonstrate strong performance empirically" (J §2).
- **Sequential.** Each subproblem is solved with the earlier ones' choices held fixed. Repeated, this is coordinate
  descent, which solves "optimization problems by successively performing approximate minimization along coordinate
  directions or coordinate hyperplanes" (J §8). Done once, in a fixed order, it is Unity's "sequentially" (J §1).
- **Across calls.** A call's measured outcome adjusts the next call's variables. This is the online method rdma/allocate.py
  uses: a projected online gradient step, as the code cites it (Zinkevich 2003).

Several objectives over the same variables can also be combined lexicographically: R5's "one lexicographic linear
program (H. Isermann, Linear lexicographic optimization, OR Spektrum 4, 1982)", as the code cites it.

## 3. The optimizers in the code, and how each composes

Each row gives:
- the function;
- its variables, which are the free choices it settles;
- its author's objective, as the code cites it;
- how it composes with the rest now.

| # | Function | Variables | Objective | Composition now |
|---|---|---|---|---|
| 1 | mesh_compile.py, sharding by `mesh_graph.propagate` | each operator's sharding | none stated: the caller's placement, propagated through each operator by its rule | before `mesh_plan`; fixed input to everything after |
| 2 | `mesh_graph.width`, called by mesh_compile.py | a chain cover of the calls' partial order | least chains: the width "by Fulkerson's matching in Hopcroft and Karp's phases" (Dilworth) | before `mesh_plan`; fixes which calls commute |
| 3 | `mesh_graph_flow`, called by `mesh_plan` | each demand's trees and their shares (`parent`, `share`) | a concurrent flow by multiplicative weights; CollectivePermute's pairs are "routed jointly by Garg and Koenemann's rounds" (mesh_compile.py) | joint over every demand. Each round's tree is a shortest-path tree under the current lengths (Dijkstra's selection, mesh-graph.c:307–320), nested inside the rounds. First of `mesh_plan`'s sequence |
| 4 | `mesh_graph_fill` | units per route (`count`) | "makespan of equal jobs on uniform machines with release times, which earliest finish solves exactly" | sequential: after row 3, with the trees fixed |
| 5 | `mesh_graph_lags` | none (computed) | least total lag over walks (Dijkstra 1959) | sequential |
| 6 | `mesh_graph_completion_times`, then `mesh_graph_heap_order` | completion times C_j, hence the transfers' order | minimize C_max, then ΣC_j (Chen 2022; Hall et al. 1997; Queyranne 1993; Isermann 1982), by HiGHS | lexicographic over two objectives; joint over every transfer on every resource (cables and devices), R4's precedence edges included; sequential after rows 3–5 |
| 7 | `mesh_graph_cycle_mean`, `mesh_graph_convert_sc`, `mesh_graph_determine_delays` | added synchronization edges and their delays | BSL: the period T = max over cycles of Σt(v)/Delay(C) (CP §4, Lemma 3) held, with the least added delays | sequential after row 6 |
| 8 | `mesh_graph_chains`, then `mesh_graph_fat_cover` | each copy's location (its colour) | fewest colours for cyclic lifetimes (Hendren et al.'s Fatcover) | joint over every value's copies; sequential after rows 6–7 |
| 9 | rdma/allocate.py | each rank's share of the next call | equal finish, the minimum of F(s) = Σ∫t_i (Beckmann 1956), by projected online gradient (Zinkevich 2003) with recursive least squares (Haykin 2014) and grain rounding (Beaumont et al. 2001) | across calls: its output is the next call's configuration |
| 10 | `mesh_graph_list_schedule`, called by rdma/mesh-flow.c | units per link within one transfer | Graham's list scheduling, so links of unequal rate "end together" | at run time, inside each transfer; not joined with `mesh_plan` |
| 11 | kernel parameters (the MPP matmul's tile constants, mesh_compile.py `_engine`) | none chosen | none | fixed per shape; not joined |

## 4. What is joint, nested and sequential today

- **Joint within one solve:**
  - the routes, over every demand (row 3);
  - the order, over every transfer and resource (row 6);
  - the locations, over every copy (row 8).
- **Nested:**
  - a shortest-path tree inside each multiplicative-weights round (row 3);
  - the list schedule inside each transfer at run time (row 10). Its objective is not part of `mesh_plan`'s.
- **Sequential, one pass, no return.** The order is rows 1 and 2, then `mesh_plan`'s rows 3, 4, 5, 6, 7 and 8.
  - A later row never revises an earlier row's choice: the routes are not revisited for the order, and the order is not
    revisited for the locations.
  - The row order is therefore an unstated priority among the authors' objectives, with sharding first and routes
    second.
- **Across calls:** the shares (row 9) reach the next call through a new compilation. data-model.md §5: "Results from
  whole calls let us adjust shares and routes for later calls."
- **Not joined:**
  - the kernel parameters (row 11);
  - the sharding (row 1), which has no objective of its own;
  - the run-time list schedule (row 10).

## 5. Methods the literature gives for making the joint choice explicit

These name the methods that exist. They are not requirements; the change requests that follow choose among them.

- **Nest the outer free choices in the inner optimizer's cost,** as Alpa does (J §2). The sharding (row 1) and the
  shares (row 9) are outer choices. Each candidate is evaluated by `mesh_plan`'s C_max (row 6) as its cost. This is
  bilevel optimization (J §7).
- **Search one representation that holds all three schedules,** as Unity's parallel computation graph does (J §1).
- **Repeat the sequence as coordinate descent** (J §8), in place of one pass. Each row is re-solved with the others
  fixed until no row's choice changes.
- **Encode routing and scheduling in one program,** as TACCL does with its MILP (J §4). TACCL also shows the
  decomposition that `mesh_plan` makes: routing first, then order, then a feasible schedule.
