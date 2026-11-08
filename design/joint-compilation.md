# The compilation's free choices, and how its optimizers compose

Quotations are from these records, each copied by script from the source's lines:
- design/joint-optimization-sources-2026-11-07.md (J);
- design/completion-and-period-sources-2026-11-07.md (CP);
- design/process-network-sources-2026-11-07.md (PN);
- design/modes-and-strategies-sources-2026-11-07.md (MS);
- design/staging-sources-2026-11-07.md (ST).

Code is cited at mesh 1d721f39 and metal-microbench e57e44a3. The references in the table's last column are the ones
the code's comments gave. Those comments are replaced by citations of this table, so the references are kept here.
Unless a record is named, a reference has not been retrieved.

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
  - An algorithm that calls a subroutine is not nested in this sense. An example is a method whose every step solves a
    shortest-path problem. Nesting needs a second optimization problem that stands as the first one's constraint.
- **Sequential.** Each subproblem is solved with the earlier ones' choices held fixed. Repeated, this is coordinate
  descent, which solves "optimization problems by successively performing approximate minimization along coordinate
  directions or coordinate hyperplanes" (J §8). Done once, in a fixed order, it is Unity's "sequentially" (J §1).
- **Across calls.** A call's measured outcome adjusts the next call's variables (row 9).

Several objectives over the same variables can also be combined lexicographically (row 8).

## 3. The optimizers in the code, and how each composes

`mesh_plan` calls its steps in this order (mesh-collective.c:2735–2760; within `plan_function`, :2500–2501):
1. `plan_function`, which computes the routes (row 3), the split (row 4) and the lags (row 5);
2. `plan_graph`, the synchronization (row 6);
3. `plan_place`, the locations (row 7);
4. `plan_schedule`, the order (row 8).

Rows 1 and 2 run in mesh_compile.py before `mesh_plan`. Row 9 runs between calls, and row 10 runs at run time.

Each row gives:
- the function;
- its variables, the free choices it settles;
- its objective, as the code states it;
- how it composes now;
- the references the code's comments gave.

| # | Function | Variables | Objective | Composition now | References |
|---|---|---|---|---|---|
| 1 | mesh_compile.py `_complete`, by `mesh_graph.propagate` | each operator's sharding | none stated: the caller's placement, propagated through each operator by its rule at two priority levels | before `mesh_plan`; fixed input to every later row | GSPMD (J §5), its sections 3.5, 4.1 and 4.3 |
| 2 | `mesh_graph.width`, called by mesh_compile.py | a chain cover of the calls' partial order | the fewest chains, equal to the width | before `mesh_plan`; fixes which calls commute | R. P. Dilworth, Annals of Mathematics 51, 1950; D. R. Fulkerson, Proc. AMS 7, 1956; J. E. Hopcroft and R. M. Karp, SIAM J. Comput. 2(4), 1973 |
| 3 | `mesh_graph_flow` | each demand's trees and their shares (`parent`, `share`) | a concurrent flow by multiplicative weights; every round computes, for each demand, a shortest-path tree under that round's lengths (mesh-graph.c:307–320) | joint over every demand; first step of `mesh_plan` | Garg and Könemann, as mesh_compile.py names the rounds |
| 4 | `mesh_graph_fill` | units per route (`count`) | earliest finish of equal units over routes with latency and rate | sequential: after row 3, its trees fixed | none given |
| 5 | `mesh_graph_lags` | none (computed) | least total lag over walks | sequential, after row 4 | E. W. Dijkstra, Numerische Mathematik 1, 1959 |
| 6 | `mesh_graph_cycle_mean`, `mesh_graph_min_delay`, `mesh_graph_convert_sc`, `mesh_graph_determine_delays` | added synchronization edges and their delays | the period T = max over cycles of Σt(v)/Delay(C) (CP §4, Lemma 3) held, with the least added delays | sequential, after row 5 | BSL 1997 §9, Figs. 6 and 8 (CP §4); E. L. Lawler, Combinatorial Optimization: Networks and Matroids, 1976 |
| 7 | `mesh_graph_chains`, then `mesh_graph_fat_cover` | each copy's location (its colour) | the fewest chains; then the fewest colours for cyclic lifetimes | joint over every value's copies; sequential, after row 6 | Dilworth 1950, Fulkerson 1956 and Hopcroft–Karp 1973 as in row 2; E. A. Dinic, Soviet Math. Doklady 11, 1970; C. Mukerji, Register allocation using cyclic interval graphs, M.Sc. thesis, McGill University, 1994, §3.2.6 and Algorithm A.0.1 |
| 8 | `mesh_graph_completion_times`, then `mesh_graph_heap_order` | completion times C_j, hence the transfers' order | minimize C_max, then ΣC_j | lexicographic over two objectives; joint over every transfer on every resource (cables and devices), R4's precedence edges included; sequential and last, after row 7 | C.-Y. Chen, arXiv 2205.02474, 2022; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3), 1997; M. Queyranne, Math. Programming 58, 1993; Q. Huangfu and J. A. J. Hall, Math. Program. Comput. 10(1), 2018 (HiGHS); H. Isermann, OR Spektrum 4, 1982; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2 |
| 9 | rdma/allocate.py | each rank's share of the next call | equal finish, the minimum of F(s) = Σ∫t_i | across calls: its output is the next call's configuration | as allocate.py gave them: Beckmann 1956; Zinkevich 2003; Haykin 2014, Table 10.1; Goodwin & Sin 1984; Beaumont et al. 2001, Alg. 3.1; Bitran & Hax 1981 |
| 10 | `mesh_graph_list_schedule`, called by rdma/mesh-flow.c | units per link within one transfer | earliest finish on links of unequal rate | a separate stage at run time, after `mesh_plan`, inside each transfer; not joined with `mesh_plan`'s objective | R. L. Graham, Bell System Technical Journal 45, 1966 |
| 11 | kernel parameters: the MPP matmul's tile constants (mesh_compile.py `_engine`) | none chosen | none | fixed per shape; not joined | Apple, MPPTensorOpsMatMul2d.h, macOS 26.4 SDK |

## 4. What is joint and sequential today

- **Joint within one solve:**
  - the routes, over every demand (row 3);
  - the locations, over every copy (row 7);
  - the order, over every transfer and resource (row 8).
- **Nested, in §2's sense:** none.
- **Sequential, one pass, no return.** Rows 1 and 2, then rows 3, 4, 5, 6, 7 and 8 in that order.
  - A later row never revises an earlier row's choice.
  - The row order is therefore an unstated priority among the objectives.
- **Across calls:** the shares (row 9) reach the next call through a new compilation. data-model.md §5: "Results from
  whole calls let us adjust shares and routes for later calls."
- **Not joined:**
  - the run-time list schedule (row 10);
  - the kernel parameters (row 11);
  - the sharding (row 1), which has no objective of its own.

## 5. Methods the literature gives for making the joint choice explicit

These name the methods that exist. They are not requirements; the change requests that follow choose among them.

- **Nest the outer free choices in the inner optimizer's cost,** as Alpa does (J §2). The sharding (row 1) and the
  shares (row 9) are outer choices. Each candidate is evaluated by row 8's C_max as its cost. This is bilevel
  optimization (J §7).
- **Search one representation that holds all three schedules,** as Unity's parallel computation graph does (J §1).
- **Repeat the sequence as coordinate descent** (J §8), in place of one pass. Each row is re-solved with the others
  fixed until no row's choice changes.
- **Encode routing and scheduling in one program,** as TACCL does with its MILP (J §4). TACCL also makes the
  decomposition `mesh_plan` makes: routing first, then order.
