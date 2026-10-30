# The collective solver: an exact packing LP and a compiled periodic schedule, 2026-10-30

From the research workflow of 2026-10-30 (a literature survey, a map of the code, a design, and an adversarial review
of it; the review's corrected design is the one below).

## Literature
The right class of solver for this fabric is a steady-state (fluid) packing LP, solved exactly, followed by a periodic page schedule fixed at compile time. Rings, double binary trees and hierarchical NVLink-then-IB schedules are not in that class. The LP and the schedule each come from published work with proofs.

(1) Routing: one LP over the real multigraph, with 13 cables as 26 directed capacities. The variables are spanning arborescences for every collective that moves data along trees (all-gather, reduce-scatter, all-reduce, broadcast, reduce) and paths for all-to-all and WRITEs. Each combining node's GPU sum rate is a capacity of its own. Merging a child's page at node v costs v's time once per child edge, so that cost becomes an edge weight and Edmonds' min-cost arborescence stays the pricing oracle. For all-reduce, each root's share and each cable's split between reduce and broadcast are variables (ForestColl Appendix G). The equal segments mesh fixes today are not.

It solves exactly in polynomial time. Two routes: max-flow binary search with Edmonds/Tarjan/Bérczi-Frank packing, or column generation priced by Edmonds' arborescence. A communicator's non-member relays are switch nodes, removed exactly by edge splitting. On this fabric the explicit LP is tiny: 24 spanning trees, so 120 rooted columns. A rational simplex solves it exactly with no ε, and every member gets the same vertex.

(2) Scheduling: the LP's rational rates become a periodic schedule (Beaumont-Legrand-Marchal-Robert steady-state scheduling; Bertsimas-Gamarnik fluid relaxation; Leighton-Maggs-Rao O(congestion+dilation)). That schedule fixes each cable's FIFO page order at compile time and shifts dependencies by one period per hop. The makespan is then the LP bound plus (tree depth + 1) periods, with no runtime arbitration. The time-expanded LP (TE-CCL's LP form) is used only for the fill and drain, and for small sizes where depth and α dominate. Once the trees are fixed, every copy is explicit, so that stays an LP.

Bounds at 64 MiB and 8.7 GB/s per cable, computed from the cable list:
- all-gather and reduce-scatter: 2.06 ms (ForestColl's cut ⋆: the four Studios against the Mini, 3 cables).
- broadcast and reduce: 2.57 ms from every root (Edmonds: each root's minimum cut is 3 cables).
- all-reduce: 2.57 ms, the cut bound for any algorithm (M bytes into and out of the Mini over 3 cables). It equals the Nash-Williams/Tutte tree-packing value of 3 cables at the partition {Mini}|rest, so reduce-then-broadcast trees with free roots reach it. That is below the 3.24 ms bound of the current trees.
- Measured: 2.7, 6.9, 4.1 and 8.1 ms.

What mesh leaves unsolved, or solves and then ignores:
- (a) mesh_solve (rdma/mesh-collective.c:831) is a multiplicative-weights loop at the constant MESH_SOLVE_EPS 0.1 (rdma/nccl-mesh.c:2681). Each round picks a Dijkstra shortest-path tree (mesh-collective.c:858). A shortest-path tree is not the least-weight tree the packing LP's dual needs, so the Garg-Könemann guarantee does not hold. mesh_arborescence, the Chu-Liu/Edmonds oracle, sits above it in the same file and is used only by mesh_trees_pack.
- (b) demands_of gives every all-reduce rank ceil(count/n) (nccl-mesh.c:2726). Each root's share should be a variable of the solve, not an input.
- (c) No node has a combine capacity.
- (d) There is no time dimension. Each cable's order is a round-major Kahn order (mesh-routed.c:332), not derived from the rates.
- (e) The default trees are capped at n per root (nccl-mesh.c:550), which the review measured at 1.67x the min cut for broadcast and reduce.
- (f) The solve runs inside plan_calls (nccl-mesh.c:2813), not as an operand kept in pages. MCCL (CoNEXT 2026) shows a column-generation solution warm-started from the previous one. With an exact deterministic LP, the no-op is the same values.
- (g) WRITE and all-to-all routes are shortest paths with ties split by hash, an equal-cost multipath assumption from Clos datacenter fabrics. The fix is a max concurrent multicommodity flow LP: 20 commodities on 26 directed cables.

Why the uniform-domain algorithms are wrong here:
- A ring makes every node send and receive 2(N-1)/N·M = 1.6M. The optimum keeps the Mini a leaf at M each way.
- The best Hamiltonian ring on this graph (u1-u2-mini-u3-u4) has two single-cable edges, so a bidirectional ring all-reduce takes about 6.2 ms against 2.57.
- Double binary trees (Sanders-Speck-Träff) assume a full-bisection switch and equal nodes. Three of the 10 node pairs have no cable here.
- Hierarchical NVLink-then-IB, BlueConnect and Blink's three-phase assume a split into equal fast domains. The cable counts (4, 3, 2, 1, 1, 1, 1) form no such hierarchy, and the binding cut is a single weak node.
- Patarasuk-Yuan's 2(N-1)/N bound is the homogeneous special case: its averaging step assumes equal processes. Its aggregate here is 2.37 ms, and it is not the binding bound.
- Trivance and Swing, which mesh cites for its single-phase all-reduce, are ring and torus algorithms.

ForestColl itself conjectures that reduce-scatter plus all-gather is enough only "with equal bandwidth per compute node". That fails here: the Mini has 3 cables and the Studios 5 or 6.

The bound numbers come from a short exact enumeration in /private/tmp/claude-501/-Users-mdot-metal-microbench/c2cfd50a-84db-442c-b354-c91c18d47940/scratchpad/calc/bounds.py. I edited nothing in mesh.

## The code as it is
Decision map of the mesh collective at HEAD 07d68ea4, read only (all paths are under /Users/mdot/dox/mesh). Only one decision is made by an optimization with a stated objective: which trees carry each segment, and how a segment is split among them. That is mesh_solve, a fixed-round multiplicative-weights solver that minimizes the most-loaded directed link's load over its capacity, solving all of a group's calls together. Even that solver has three defects. (1) For every collective demand with all nodes as leaves, its oracle is Dijkstra's shortest-path tree, not the least-cost tree. The exact oracle, mesh_arborescence (Chu-Liu/Edmonds), sits right above mesh_solve unused; only mesh_trees_pack calls it, and that runs only for configured forms and is capped at `most` trees per root. (2) Its inputs are fixed rules: all-reduce segments are equal, the owner of segment r is rank r, and the all-reduce is always two-phase on the same tree. (3) Its objective is only the fluid cable bound, with no FIFO effects, tree depth, combine compute or relay cost. Every timing decision is a fixed rule: FIFO order per cable (round-major Kahn order by lowest index), call-major order of a group's calls, consumption order, pipelining granularity (page, untiled pieces, 4x1024 GPU chunk), landing gate, staging, and whether calls overlap (they never do: each launch completes itself, GENERATIONS=2, queue barriers). Solvers that exist but are ignored or used on another subproblem: mesh_write_order (a compile-time list scheduler with a cable model; only all-WRITE groups use it), mesh_arborescence and mesh_trees_pack (correct oracle, unused by default), routing.Fabric.allocate (exact HiGHS min-congestion LP, used only for pair routes when configured; in torch_mesh bind it re-routes the tree edges' own pair demands over relays, a second solve sequential to the tree solve), and allocate.min_max/equal_finish (exact min-max allocation, unused for segment sizes). Node compute rates enter no compile decision; the only node facts are the runtime worker count (cores) and threadgroup width. Routing is solved synchronously inside plan_calls into heap state and in-place weight writes, not as a k-buffered operand in region pages. Measured against bounds (64 MiB): all-reduce 8.1 vs 3.24 ms, reduce-scatter 6.9 vs 2.06, broadcast 4.1 vs 2.57, all-gather 2.7 vs 2.06, exchange 5.4 vs 2.57. Per the review's simulation, the tree and ordering defects alone account for most of the reduction gap: tree choice, FIFO order and the combined-forward refine pass together bring the simulated all-reduce from 4.89 to 3.28 ms against the 3.24 bound. Literature named in `gaps` is the class each decision belongs to; it is a pointer for the next stage, not something verified here.

## The design
**Choice.** **There is one solver.** mesh_solve becomes the exact steady-state packing LP over the communicator's directed pair links, solved by column generation:
- **Master problem:** a dense simplex with a lexicographic ratio test.
- **Pricing:** there are three exact oracles. Which one applies follows from each demand's own bytes, not from its collective kind:
  - **A tree demand whose up ≠ down** (broadcast, reduce, one all-gather or reduce-scatter segment) uses Chu-Liu/Edmonds at its root (mesh_arborescence, already in the file).
  - **A tree demand whose up = down** (the all-reduce) uses a minimum spanning tree (Kruskal) on π(u→v) + π(v→u). Such a tree loads both arcs of every edge equally whatever its root, so the root is no LP variable. After the solve, each column is rooted at its tree's center (least height). Ties go to the node with more pool workers (mesh_gpu_cores), then to the lower rank.
  - **A pair demand** (WRITE, all-to-all) uses Dijkstra.

**Lexicographic phases on the optimal face.** Each phase reuses the same columns and adds only per-arc pricing terms.
- Phase 1 minimises the makespan T.
- Phase 2 holds T* and minimises the largest combine work per pool worker over the nodes.
- Phase 3 holds both and minimises the hop excess, in page times.

**Realization.**
- Each pair's column loads are filled onto that pair's cables in column order, up to each cable's capacity × T*. This is McNaughton's wrap-around rule: at most m−1 split edges a pair. Today every tree edge is split over every cable of its pair.
- mesh_compile_sets' coalescing is deleted, so one entry is one column's edge on one cable.
- Combining relays forward per unit.
- One list schedule orders every cable's FIFO and every channel's consumption, with IEEE-deterministic arithmetic.

**Solutions are operands.**
- Rows live in region leases, GENERATIONS-buffered. Each request is solved on its own thread, never in planning or in a call.
- A demand list never seen before compiles at once from its calls' init rows, which are feasible.
- It takes its exact row at its next declaration. Every rank agrees on that switch point.

**What is deleted:** rings, trees, packers, forms, installed routes and every-path weights. So are the table-priced combine rows, the all-reduce root variables and the even all-to-all init row that the earlier draft proposed. Caller-supplied programs and paths at ncclMeshCommInitRank stay as the explicit-operand route.

**Why.** **The LP's certificate covers the binding cut, and its direction differs by kind.**
- I re-solved explicit-column LPs on this map with scipy HiGHS (scripts in /private/tmp/claude-501/-Users-mdot-metal-microbench/c2cfd50a-84db-442c-b354-c91c18d47940/scratchpad/review/).
- All-gather (2.057 ms): positive duals sit on the Mini's 3 inbound cables.
- Reduce-scatter (2.057 ms): positive duals sit on the Mini's 3 outbound cables, not its inbound ones as the draft says.
- All-reduce (2.571 ms): both directions are binding.
- The Studios' inbound cables are not saturated at any T*-optimal steady state, so "every inbound cable of every hungry node saturated" is certified only on the binding cut. The draft read it as certified on every hungry node.

**The draft's all-reduce root variables do nothing, and its claim that the Mini combines nothing is luck.**
- With up = down, a tree's link loads do not depend on its root.
- Rooting every tree at the Mini still reaches 2.571 ms. At that optimal vertex the Mini combines a whole M (5× the equal-segment form's 0.2·M).
- The draft's phase 2 counts arcs and cannot tell these vertices apart.
- Pricing undirected spanning trees by MST removes the y variables and the group rows. Rooting each tree at its center then minimises its height exactly, which is better than the arc-count proxy.
- On this map all 24 spanning trees have center height 2.
- At T* the Mini is a leaf of every optimal tree (cut {Mini} = 3 cables = the packing value). It is therefore never a center and combines nothing, as a property of the rule rather than of the vertex the simplex lands on.

**The weak node's GPU work is unconstrained on the optimal face.**
- At the reduce-scatter's T*, the Mini combines between 0.2·M and 0.8·M depending on the vertex: 13.4 to 53.7 MB a call at 64 MiB, on its 16-core pool.
- Phase 2 (least largest combine per pool worker) pins it at 0.2·M. Phase 3 then gives mean height 2.2, max 3.
- For the all-gather, phase 3 gives depth 2.
- The draft's combine-rate rows instead priced the Mini from a memory-bandwidth table / 3. That is inert on this map and models the wrong limit: the combine reads 32-bit volatile system-coherent words (finding P5), not DRAM.

**FIFO realization.**
- The steady-state T* is a fluid bound. Whole-entry FIFO cables reach it only where few columns share a cable.
- Splitting every tree edge over every parallel cable by bandwidth (mesh_write_routes_compose) does the opposite: every cable of a pair time-shares every tree.
- Sequential fill to capacity × T* is exact and splits at most m−1 edges a pair.
- On this map there is an integral form: 3 cable-disjoint spanning trees (3 families) carry the all-reduce at 2.571 ms, M/3 each, rooted at u1, u1, u2, height 2. Each cable then carries one stream in each direction and needs no order. When the simplex returns that vertex, the fill realizes it exactly.

**Exactness and time, stated as they hold.**
- Column generation with a lexicographic rule is finite and deterministic but is not polynomial. The polynomial bound (Grötschel-Lovász-Schrijver) is the ellipsoid method's, given an exact separation oracle.
- The all-reduce's same-tree class is optimal here: 2.571 ms equals the information bound, since each node must send M and receive M, and the Mini has 3 cables.
- It is not optimal in general. ForestColl §5.7 states that combining reduce-scatter and all-gather trees "does not guarantee allreduce optimality", and its Appendix G gives an LP with separate trees for that.
- The table of alternatives (Garg-Könemann approximate; TE-CCL and SCCL exponential; TACCL sketches; rings/trees/hierarchies with uniform domains) stands.

**Exchange numbers.**
- The benchmark's exchange is a cyclic shift (torch_mesh/benchmarks.py exchange_counts), not M/4 a pair.
- Its multicommodity-flow (MCF) optimum is 2.571 ms with ranks in node order u1, u2, u3, u4, Mini, and 3.085 or 3.857 ms for other rank orders.
- 2.893 ms is the even all-to-all's optimum.
- The review's 2.70 ms routes are therefore 5% above the optimum for the node-order case. The measured 5.4 ms is still mostly not routing.

**Re-checked from the draft and confirmed:**
- 24 spanning trees, 120 rooted arborescences.
- Broadcast and reduce reach 2.571 ms from every root.
- The all-gather and reduce-scatter bounds are 2.057 ms.

**Formulation.** **Sets.**
- **Directed pair links A:** capacity c_a is the sum of the pair's cable bandwidths (mesh_solve_links), divided by the least capacity. This map has 14.
- **Cables per pair, m_a:** used only by the fill.
- **Demands J (demands_of):**
  - A **tree demand** has a root r_j, leaves = every node, up_j and down_j.
  - A **free tree demand** has root UINT32_MAX and up_j = down_j. The all-reduce is one such demand.
  - A **pair demand** has a source ℓ_j and a destination r_j.
  - Row key: bytes divided by the list's gcd, kept as integers. Numerics: bytes scaled by the largest demand.
- **Columns K_j:**
  - Tree demand: spanning arborescences at r_j, carrying up_j child→parent and down_j parent→child.
  - Free demand: undirected spanning trees, carrying up_j on both arcs of each edge.
  - Pair demand: paths. α_ka is column k's per-unit load on link a.
- **Workers:** w_v = mesh_gpu_cores of rank v's node, gathered in the init's first bootstrap report.
- **Combine count:** β_kv is v's children in k (for a free tree, deg_k(v), which decomposes per edge).

**Phase 1.** Minimise T subject to:
- Σ_k α_ka x_k ≤ c_a T for each a in A.
- Σ_{k∈K_j} x_k = 1 for each j.
- x, T ≥ 0.

**Phase 1 dual.** Maximise Σσ_j subject to:
- Σ_a c_a π_a = 1.
- Σ_a α_ka π_a ≥ σ_j for every column k of j.
- π ≥ 0.

π_a > 0 marks a saturated link. Termination at T* = Σσ_j is the certificate.

**Phase 2.**
- Add T ≤ T*.
- Add a variable W with rows Σ_k β_kv x_k (bytes) ≤ w_v W for every node v.
- Minimise W.
- Pricing adds κ_p / w_p on each child arc into p, or κ_u + κ_v on a free edge. Edmonds and Kruskal stay exact.

**Phase 3.**
- Add W ≤ W*.
- Minimise Σ_k η_k x_k, where η_k = Σ over arcs p→v of k with hop_r(v) ≠ hop_r(p)+1 of page/c_a (seconds of fill).
- Free trees have no η: they are rooted at their center afterwards, which minimises their height exactly.

**Pricing.**
- **Tree demand:** mesh_arborescence on arcs p→v, weight up_j·π(v→p) + down_j·π(p→v), plus the phase terms.
- **Free demand:** Kruskal on edges u–v, weight up_j·(π(u→v) + π(v→u)), plus the phase terms. Ties go to the lower cable index.
- **Pair demand:** Dijkstra on π.
- **Entry:** most negative reduced cost below −τ, with τ = R·DBL_EPSILON·max|coefficient|. Ties go to the lowest demand, then the lowest column.

**Room.**
- Rows R = 2E + |J| + |V| + 2, which bounds the positive columns.
- The process is finite under the lexicographic rule, with no polynomial pivot bound.
- Arithmetic is IEEE +, −, ×, ÷ only, with -ffp-contract=off on both libraries. C Annex F then gives every member the same bits.

**From solution to program.**
- **Free-tree roots:** each tree's root is its center, ties broken by more w_v, then lower rank.
- **All-reduce segment cut:** root r's segment is Σ of its trees' shares × units, in page quanta: the floor, plus remainder quanta given one at a time to the root whose columns' most-loaded cable stays lowest (heterogeneity.md R5).
- **Parts within a segment:** the same rule replaces mesh_program_ranges' by-index remainder.
- **Cable fill:** for each pair, columns in index order fill its cables (index order, page-quantum boundaries) up to capacity × T* × call bytes. Each column edge becomes one or two transactions, each on a named cable.
- **Relays:** COPY and REDUCE relays forward per unit.

**Order.**
- One non-delay list schedule (mesh_write_order generalised to parent lists):
  - Each directed cable is a server at its bandwidth.
  - A per-unit forward is released at the latest first page of all its streamed parents. That is every chained COPY/REDUCE of its range, not only the last.
  - It finishes no earlier than the parents' last finish plus one page on its own cable.
  - A whole dependency is released at its parent's finish.
- Priority is release time, then production order, then upward rank, then event index.
- dependency_order's Kahn pass takes ready events by predicted time, then by event index.

**Inputs.** - **The links:** comm->map's member links, each cable at mesh_map's bandwidth (the lesser of both ends' records). Pairs are summed for the LP (mesh_solve_links) and kept per cable for the fill.
- **The page:** comm->page (block × pgsz, 16 KiB here). It is the pipelining unit, the cut quantum and phase 3's per-hop cost.
- **The calls:** each call's kind, root, counts, matrix and segments (demands_of), with a group's calls together.
- **Workers:** each rank's mesh_gpu_cores, carried in the init's first bootstrap report, so every rank holds the same vector. It only breaks ties among T*-optimal solutions.

Nothing else: no ε, rounds, most, below, shares, chunk counts, route-weight bits, capacity tables or memory rates. τ = R·DBL_EPSILON·max|coefficient| is the simplex's stated zero test.

**Outputs as operands.** **struct mesh_solution rows** (mesh-plan.h), packed in a generation's region lease, held as an ncclMeshPageTable. Each row has:
- **A header:** nodes, links, demands, columns; T*; the dual bound; W*; the phase-3 value; the binding directed links (dual > 0).
- **The integer demand key**, as page-safe struct mesh_demand {root (UINT32_MAX for a free tree), pad, leaf (UINT32_MAX for every node), pad, up, down}.
- **Columns** {demand, root, height, share}. A free tree's root is its center.
- **Parents:** n per column; a path's off-path nodes are UINT32_MAX.

**The index:** GENERATIONS page-table generations of {count, row offsets}; comm->solved names the one in place.

**Derived per call at compile, not stored.** Each is a pure integer function of the row, the call's bytes and the page:
- the all-reduce's segment cut and each segment's part cut (R5 rounding);
- each pair's cable fill (McNaughton);
- each pair's mesh_paths;
- the list schedule's publication and consumption ordinals.

**Asynchrony.** **Where solving happens.**
- Solving happens where its inputs first exist, and never on a path that waits for it unless a dependency was deduced.
- **At init**, on the init thread, after fabric_realize: the rows of every kind made without counts.
- **At a declaration whose demand list has no row:** the declaration compiles at once from its calls' init rows and starts one thread that solves that list. The thread ends when its row is published. There is no resident thread and no idle wait.

**When a row is used.** A declaration uses a list's exact row only if that list was requested at an earlier declaration.
- Declarations are SPMD, so every rank switches at the same declaration whatever its local solve time.
- If that later declaration arrives before the local solve has published, it joins the solve's thread under MESH_PLATFORM_NS. That is the one wait, deduced from the row being an input of the compile.

**k-buffering.**
- GENERATIONS = 2 page-table generations.
- A solve writes the generation not in place, then stores comm->solved with release under group_lock and frees the replaced generation. Planners read only under group_lock.
- Rows are only added, never changed.
- A re-declaration with the same key reads the same values from the same pages: the no-op.

**No coordination between members.**
- Each row's contents are a pure function of the gathered map, the gathered worker counts and the integer demand key.
- The arithmetic is IEEE, uncontracted, and uses no libm (C Annex F).
- No solution crosses the fabric, and no member waits on another.

**Deletions.** **rdma/mesh-collective.c**
- mesh_solve's multiplicative-weights body (rounds, width, exp lengths, the shortest-path tree, hits) and mesh_solve_room's eps form.
- mesh_trees_pack, struct mesh_packed and mesh_packed_heavier.
- mesh_compile_sets' message coalescing loop.
- mesh_compile_sets' `parts==before` exit for the all-reduce.
- mesh_program_ranges' by-index remainder.
- mesh_group_lay's call-major loop.

**rdma/nccl-mesh.c**
- MESH_SOLVE_EPS; struct flow, flow_solve and flow_free.
- demands_of's n ceil(count/n) all-reduce demands.
- writes_weigh.
- comm->writes' every-path default, write_overrides and ncclMeshCommWriteRoutes.
- compile_form; comm_routes' packed branch; comm->trees, rooted, forms and form_count; ncclMeshCommForms.
- program_of's `below`.
- The `call` key in consumption_order_compare.
- order_writes' private scheduler.

**rdma/mesh-routed.c:** the lowest-index Kahn scan; refine_pieces' whole-rank skip.

**rdma/mesh-fabric.c:**
- mesh_fabric_refine's COPY-only owner and its off-quantum -EINVAL.
- mesh_write_routes_default.
- mesh_write_routes_compose's routes and overrides parameters, and its split of tree transactions over every cable by bandwidth.

**rdma/mesh-write-order.h:** the single-parent task.

**Headers, front ends and docs:**
- nccl.h: ncclMeshForm, ncclMeshTrees, and the forms and write-routes declarations.
- mesh-plan.h: mesh_trees_pack and mesh_demand's `leaves` pointer.
- routing.py forms(); configuration.py's forms parameter and call; the forms arguments in examples/chain-service.py and lstm-service.py.
- mesh_rank.m install_forms and install_weights.
- NCCL.md's Forms section.

**Dropped from the reviewed draft, never added:**
- the all-reduce's y variables and group rows;
- the combine-rate rows with ρ_v from etc/mesh-capacity.json;
- the rank-0 solve-and-gather fallback;
- the even all-to-all init row;
- the synchronous solve inside definition_prepare;
- the deletion of ncclMeshCommWritePaths, which mesh_rank.m routing_report reads at 886.

## Steps

1. **mesh_solve becomes the exact packing LP with three exact oracles, and both libraries are built without FP contraction** (rdma/mesh-collective.c (mesh_solve 831-901, mesh_solve_room 824-829; a static simplex and a Kruskal pricer beside mesh_arborescence 748-791 and mesh_solve_links 803-822); rdma/mesh-plan.h (141-155: struct mesh_demand's root UINT32_MAX means a free tree; declarations and comment); rdma/nccl-mesh.c (MESH_SOLVE_EPS 2675, flow_solve 2678-2690); rdma/Makefile (-ffp-contract=off on the libmesh rule at 152 and the libnccl-mesh rule at 135). Line numbers as of mesh ad4a8bdd.). **Signatures.**
- `uint32_t mesh_solve(const struct mesh_link_map *, const uint32_t *workers, const struct mesh_demand *, uint32_t count, uint32_t room, uint32_t *demand, uint32_t *root, uint32_t *parent, double *share)` (no eps).
- `mesh_solve_room` returns 2E + count + nodes + 2.

**Master problem.**
- Variables: x_k and T. Rows: link rows Σα_ka x_k − c_a T ≤ 0, and one demand row Σ_{k∈j} x_k = 1 per demand.
- Phase I uses artificials on the demand rows.
- Dense basis inverse; lexicographic ratio test.

**Pricing, by the demand's own data.**
- A rooted tree (leaves NULL, up ≠ down, or a root given): mesh_arborescence over the 2E arcs.
- root == UINT32_MAX with up == down: Kruskal over the E pairs with weight up·(π(u→v)+π(v→u)). Ties go to the lower link index.
- A single leaf: Dijkstra.

**Output.** The positive columns. A free tree's root is written by step 2.

**Delete.** The rounds, width, exp lengths, the per-round shortest-path tree, hits and MESH_SOLVE_EPS.

**Determinism.** -ffp-contract=off goes in the Makefile because only mesh-collective.c carries `#pragma STDC FP_CONTRACT OFF`. mesh-routed.c and nccl-mesh.c (which includes mesh-write-order.h) compute the doubles that decide each cable's FIFO order on both of its ends.

**Docs.** Cite Beaumont et al. 2005, Edmonds 1967, Kruskal 1956 and Nash-Williams/Tutte 1961, and ForestColl 2024 §5.7 and App. G for the all-reduce's limits. Expected: LP bound at 64 MiB:
- Broadcast and reduce: 2.95-3.14 ms (u1-u4) and 7.71 (Mini root) go to 2.571 from every root.
- All-reduce: 3.24-3.66 goes to 2.571.
- Reduce-scatter and all-gather: 2.07 goes to 2.057.

Other numbers:
- The all-reduce row has 15 + 1 rows instead of 14 + 5 + 6.
- The review's page simulation with arborescence trees: broadcast from u1 5.25 → 3.09 ms.
2. **Lexicographic phases 2-3 on the optimal face; all-reduce trees rooted at their centers** (rdma/mesh-collective.c (mesh_solve: the W variable and node rows, the depth phase, center rooting); rdma/mesh-plan.h (comment); rdma/nccl-mesh.c (the init report carries each rank's mesh_gpu_cores: struct report 986-990, fabric_discover 999ff)). **Phase 2.**
- Bound T ≤ T*. Add W with rows Σ_k β_kv x_k ≤ w_v W, and minimise W.
- Duals κ_v become per-arc pricing terms: κ_p / w_p on a child arc into p, κ_u + κ_v on a free edge.

**Phase 3.**
- Bound W ≤ W* and minimise Σ η_k x_k, with η in page/c_a seconds per non-shortening arc.
- Free trees have no phase 3.

**Rooting.** Each free tree is rooted at its center, ties to the larger w_v, then the lower rank. Record each column's root and height.

**The workers operand.** w_v is the pool's own size (d946e1e4: at most one worker a core). It is a hardware fact read once, gathered so that every rank holds the same vector. It only breaks ties among T*-optimal solutions, so it can never cost link time. A rank whose registry states none reports 1. Expected: Measured with explicit LPs on this map:
- **All-reduce:** every column has height 2. The Mini is a leaf of every T*-optimal tree and never a center, so it combines 0 (versus 0.2·M with equal segments and up to 1.0·M on an unconstrained optimal vertex).
- **Reduce-scatter:** the Mini combines 0.2·M (13.4 MB at 64 MiB), against up to 0.8·M on an unconstrained vertex. Heights are mean 2.2, max 3.
- **All-gather:** depth 2.
- **Broadcast:** depth 3 from the Studios.

The page here is 16 KiB at 1.9 µs a hop, so depth is a small share at 64 MiB and a larger one from 64 KiB to 1 MiB.
3. **The all-reduce is one free demand; its segments come from the tree roots, with R5 integer rounding** (rdma/nccl-mesh.c (demands_of 2692-2726: one demand {UINT32_MAX, up = down = bytes}, deleting `(c->count + n - 1) / n`; plan_calls 2793ff: the all-reduce passes a cut; solved_steps 2728ff); rdma/mesh-collective.c (mesh_compile_sets 401-535: an all-reduce keeps n segment indices and accepts a zero-length segment for a node that roots no tree, replacing the `parts==before` exit at 483 for the all-reduce; mesh_program_ranges 582-612: R5 remainders)). **Cut.**
- plan_calls cuts the operand by root r's Σ shares × units in page quanta.
- Remainder quanta go one at a time to the segment whose columns' most-loaded cable, at the call's integer loads, stays lowest (heterogeneity.md R5, Ibaraki-Katoh greedy).
- The cut reaches collective_steps as the all-reduce's `cut`. The program keeps n segment indices, which collective_steps already checks.
- mesh_program_ranges' within-segment remainder uses the same rule instead of "the first parts".

**Program form.** One tree still carries a root's reduce and broadcast. That is exact for this class, and optimal here because 2.571 ms equals the information bound. Expected: - T* stays 2.571 ms.
- Segments land at Studio roots only, for example 2/3 at u1 and 1/3 at u2 on the integral vertex. Zero-length segments go to u3, u4 and the Mini.
- A small call's quanta go to the columns whose cables are least loaded instead of by index.
4. **Combining relays forward per unit, and one entry is one column's edge** (rdma/mesh-fabric.c (mesh_fabric_refine 28-98: the owner test, the -EINVAL exits); rdma/mesh-routed.c (refine_pieces 55-110: the whole-rank skip on -EINVAL); rdma/mesh-collective.c (mesh_compile_sets' coalescing loop from 505: deleted, iterating `messages`, as 589096e1's hunk did); rdma/mesh-plan.h (refine comment 181-196)). - The owner of an interval becomes the latest overlapping receive, COPY or REDUCE (`next=j` for both).
- An off-quantum owner change keeps that SEND's whole-publication dependency instead of returning -EINVAL.
- Delete refine_pieces' whole-rank skip.
- Delete coalescing. A coalesced SEND spans several columns' edges with different sources, which contradicts the schedule's premise that an entry is one column's edge. Review finding P2 measured the loss.
- Off-quantum owner changes are then reachable only at caller-forced segment boundaries, and no step spans two segments (mesh_program_steps_cut).
- send_dependencies' `same` unit check stays. It runs at bind, not per unit. Expected: The review's simulation: reduce from u1 5.13 → 4.30 ms on this alone. 136 MiB of reduce-scatter forwards and 392 MiB of all-reduce forwards stream per unit instead of waiting 0.06-0.56 ms a hop.
5. **Parallel cables are filled per pair (McNaughton), not split per edge** (rdma/mesh-fabric.c (mesh_write_routes_compose 314-357: its adjacent-pair branch no longer splits tree transactions); rdma/mesh-routed.c (mesh_routed_compile 410ff: an optional per-transaction paths array beside the per-pair table); rdma/nccl-mesh.c (bind_routed_steps 2555ff passes it; plan_calls makes the fill)). **The fill.**
- For each pair, take the call's columns crossing it in column order.
- Fill each cable in index order, up to its capacity × T* × the call's bytes, at page-quantum boundaries.
- A column edge crossing a fill boundary becomes two transactions, each with a one-cable path. Every other column edge is one transaction on one cable.
- This is McNaughton's wrap-around rule (1959). The bytes have no time-overlap constraint, so the same sequential fill is exact for cables of unequal bandwidth.

**Pair demands.** Their paths take the same fill per hop.

**Determinism.** Everything is integers in column order, so both ends make the same transactions. Expected: - A pair's cable loads stay within one page of capacity × T*.
- Each column edge uses 1-2 cables instead of all m (u3-u4: 4, u1-u2: 3). Pieces and entries on multi-cable pairs fall by up to m×.
- On the all-reduce's integral vertex (3 trees × 1/3), each cable carries exactly one tree each way: 24 entries a call, with no FIFO choice left.
6. **One list schedule orders every cable and every consumption** (rdma/mesh-write-order.h (all: a parent list per task); rdma/mesh-routed.c (dependency_order 332-398; mesh_routed_compile passes cable capacities); rdma/mesh-plan.h (an inline `mesh_step_forwards`); rdma/nccl-mesh.c (forwards 2904 and order_writes 3013ff call the shared pieces)). **mesh_write_task.** Takes a parent list {task, streamed}.
- Release: max over parents of (streamed: start + first page / the parent's arc rate; whole: finish).
- Finish: max(start + bytes/rate, streamed parents' finish + first page / own rate).

**Streamed parents.** Every prior COPY or REDUCE of the forwarded range that the runtime chains chunk by chunk (f1a821ab's metal_after). This covers all of a combining relay's children, not only the last receive. Any other prior is whole.

**Kahn pass.** Keep the edge set and the single Kahn pass, with a ready heap keyed by predicted time, then event index. Delete the lowest-index scan.

**WRITE groups.** order_writes keeps its fragment table and calls the same scheduler.

**Determinism.** All arithmetic is in TUs built with -ffp-contract=off (step 1). Expected: The review's simulation:
- All-reduce 4.89 → 4.25 ms alone, 3.28 with arborescence trees.
- Reduce-scatter 2.82 → 2.61 alone, 2.07 with steps 1-4.

With steps 1-5, the target is 2.571 / 2.057 plus a few page times of fill. The measured 8.1 / 6.9 ms also carry the call boundary (about 1 ms), whole-consumption waits and per-claim GPU costs.
7. **A group is one transaction graph, ordered jointly** (rdma/nccl-mesh.c (plan_calls, bind_routed_steps 2555ff, schedule_group 2914ff, consumption_order_compare 2858, binding_relations 4493); rdma/mesh-collective.c (mesh_group_lay 1052-1062); rdma/mesh-plan.h (mesh_group* 336-346)). - One mesh_routed_compile lowers a group's routed calls together. Each call keeps its own buffer numbers, except where binding_relations finds overlapping storage.
- Step 6's schedule orders the whole group, matching its joint solve.
- Delete mesh_group_lay's call-major loop and the `call` key of consumption_order_compare. Expected: A group's calls share every cable at once. The single-call benchmark cases are unchanged.
8. **The solution is a k-buffered operand, solved off the planning path, with a switch point every rank agrees on** (rdma/mesh-plan.h (page-safe struct mesh_demand {root, pad, leaf, pad, up, down} with `leaf` replacing the `leaves` pointer; struct mesh_column {demand, root, height, share}; struct mesh_solution); rdma/nccl-mesh.c (struct ncclComm 172ff: solution[GENERATIONS], solved, requests; fabric_realize 1410ff: leases and the init rows; definition_prepare 5340 and plan_calls: hit, fallback, or join; struct flow, flow_solve, flow_free deleted; comm_free joins outstanding solves); design/fake-guards-2026-10-30.md (row "routes solved inside planning"); rdma/NCCL.md). **Rows.**
- Rows live in the generation's ncclMemAlloc lease, wrapped as an ncclMeshPageTable (the existing page-table object).
- GENERATIONS = 2 such tables; comm->solved names the one in place.
- A row holds:
  - a header: counts; T*; the dual bound; W*; the phase-3 value; the binding links;
  - the integer demand key;
  - the columns;
  - the parents.

**Init rows.** Solved on the init thread before init returns: the all-reduce, the equal reduce-scatter and all-gather, each root's reduce and broadcast, and each pair's single WRITE (a max flow). No even all-to-all row: that guesses a demand.

**Declaration of a demand list L** (under group_lock, as planning already is):
- **A row for L is in place:** compile from it.
- **L was requested at an earlier declaration on this communicator:** join that request's thread within MESH_PLATFORM_NS (an expiry as every bounded wait), then compile from its row. This dependency is deduced. It occurs only when the program re-declares faster than the solve.
- **Otherwise:** compile now from each call's init row (feasible: those trees and paths carry any counts), and start one thread that solves L.

Declarations are SPMD, so "requested earlier" is the same fact on every rank. The local solve's timing never selects a row.

**Publication.**
- The thread writes the generation not in place (the rows in place plus L's).
- It takes group_lock only to store comm->solved with release and free the replaced generation. Planners read only under group_lock, so nothing can hold the freed generation.
- Rows are only added, never changed.

**Rule.** Calls and planning never solve. Expected: - Call times are unchanged.
- A re-declaration with the same key (partition._replan, a Balancer step that keeps its parts) reads the same values from the same pages and runs no solve.
- A first ragged declaration runs at once on feasible routes. Its next declaration runs on the exact row.
- This closes the fake-guards row "routes solved inside planning".
9. **WRITE and all-to-all paths are the solution's path columns** (rdma/nccl-mesh.c (writes_weigh 2748ff deleted; ncclMeshCommWriteRoutes 650 and write_overrides deleted; ncclMeshCommWritePaths 693 kept as a reader of the in-place row's paths; comm->writes' every-path default deleted; fabric_discover 1054-1055); rdma/mesh-fabric.c (mesh_write_routes_default 256ff deleted; mesh_write_routes_compose loses `routes` and `overrides`); rdma/nccl.h (994-1001); metal-microbench mesh_rank.m (install_weights 852-869 deleted; routing_report 871ff keeps its ncclMeshCommWritePaths read)). - Each pair column becomes mesh_paths entries over the step-5 fill of each hop's cables.
- Weights are integers at the pair's route_units resolution, not at a fixed 2^40.
- routing_report keeps logging the solved paths: an observation, not an input. Expected: - Routes become the multicommodity-flow optimum for the declared matrix.
- For the benchmark's cyclic exchange that is 2.571 ms with ranks in node order u1, u2, u3, u4, Mini, and 3.085 or 3.857 ms for other rank orders. Rank order is placement, an operand, not the solver's.
- 2.893 ms is the even all-to-all's optimum.
- The review's 2.70 ms routes improve by about 5% (node order). The measured 5.4 ms is relay pacing and call boundaries.
- The factorial every-path enumeration cannot come back.
10. **One routing: configured forms and the second tree packer deleted** (rdma/mesh-collective.c (mesh_trees_pack 915-1019, struct mesh_packed, mesh_packed_heavier); rdma/mesh-plan.h (129-140, 155); rdma/nccl-mesh.c (compile_form 467ff, comm_routes' packed branch 545-563, ncclMeshCommForms 622ff, comm->trees, rooted, forms, form_count, program_of's `below` 742-749); rdma/nccl.h (ncclMeshForm, ncclMeshTrees 962-994); rdma/routing.py (forms); rdma/torch-mesh/torch_mesh/configuration.py (the forms parameter at 17 and the call at 61); examples/chain-service.py:61 and examples/lstm-service.py:54; metal-microbench mesh_rank.m (install_forms 825-850 and its call at 998); rdma/NCCL.md (Forms as configuration)). - A caller's programs and paths at ncclMeshCommInitRank stay, as the explicit-operand route.
- Every other collective is the solver's row. Expected: - The benchmark names no forms, so it is unchanged.
- One solver is left.

## Risks and the review's changes
**What I changed in the draft, and why:**
- **The certificate's direction.** The reduce-scatter's binding links are the Mini's outbound cables, not its inbound ones (HiGHS duals: mini→u1, u2, u3). The all-reduce binds both ways. The goal of saturating every hungry node's inbound cables is certified only on the binding cut. The Studios' inbound cables are not saturated at any T*-optimal steady state.
- **The all-reduce's y variables and group rows were replaced by MST pricing and center roots.** With up = down, link loads do not depend on the root: all trees rooted at the Mini still give 2.571 ms, with the Mini combining a whole M. The draft's result that the Mini combines nothing depended on which vertex the solver returned. Center rooting minimises the height exactly; every tree here has height 2, and the Mini, a leaf of every optimal tree, is never a center.
- **Phase 2 is now the least largest combine per pool worker, and the depth phase moves to phase 3, in page-time units.** The reduce-scatter's optimal face lets the Mini combine anywhere from 0.2·M to 0.8·M; phase 2 pins 0.2·M.
- **The table-priced combine rows (draft step 10) were deleted.** memory/3 from a table is a price (D6), it is inert here, and the combine is bound by system-coherent 32-bit loads (P5), not DRAM. The worker count only breaks ties on the optimal face, so it cannot cost link time.
- **Each pair's parallel cables are filled per column (McNaughton) instead of splitting every edge over every cable.** It is exact at T* with at most m−1 splits, has up to m× fewer pieces, and leaves less FIFO time-sharing. On the integral all-reduce vertex it gives one stream per cable.
- **Coalescing in mesh_compile_sets was deleted.** A coalesced entry is several columns' edges with different sources. The draft's premise that an entry is one column's tree edge was false while coalescing stayed.
- **Every chained child REDUCE is a streamed parent of a combining relay's forward**, not only the last receive. The draft's rule mis-modelled every combining hop.
- **-ffp-contract=off goes on both libraries.** Only mesh-collective.c has the pragma, and both ends of a cable compute its FIFO order from doubles. With contraction off, Annex F gives identical bits on any conforming compiler, so the rank-0 gather fallback (a peer exchange, C7) was dropped.
- **Solves left the planning path.** The draft solved on a miss inside definition_prepare. That is the fake-guards row it claimed to close.
- **The async rule makes every rank switch at the same declaration.** A naive async swap would let ranks with faster solves compile other rows, giving mismatched wire order with no runtime check to catch it.
- **The old generation is freed under group_lock.** The draft freed it immediately while planners read without the solver's lock.
- **The even all-to-all init row and its 2.89 ms claim were removed.** The benchmark's exchange is a cyclic shift: 2.571 ms in node order, 3.085 or 3.857 ms in other rank orders.
- **ncclMeshCommWritePaths stays as a reader**, because mesh_rank.m routing_report calls it.
- **The claims were narrowed:**
  - Column generation is finite, not polynomial; the polynomial bound is the ellipsoid method's.
  - The same-tree all-reduce is optimal here (it meets the information bound) but not in general (ForestColl §5.7; its App. G LP is the exact class).

**Remaining risks:**
- **Other runtime stalls bound the result.** The P1-P2 runtime stalls still cap measured times: the barrier between calls, the landing gate, whole-consumption priors, per-claim GPU costs and the roughly 1 ms call boundary.
- **No worst-case bound on the order.** A whole-entry FIFO list schedule has none where columns time-share a cable. The fill reduces time-sharing but does not remove it for the all-gather and reduce-scatter.
- **An integral cable-disjoint packing is not computed.** One exists here (3 trees for the all-reduce: Nash-Williams/Tutte; and for broadcast: Edmonds). The simplex may return a fractional vertex instead. Matroid partition (Roskind-Tarjan) would find the integral packing directly, but only where 1/T* is an integer in cable units, so it would add a branch.
- **A definition never re-declared keeps its first-declaration row**: feasible, not optimal.
- **ncclMeshCommPlan's timing changes.** A declaration that repeats a pending request joins a thread, bounded by MESH_PLATFORM_NS.
- **Small calls.** Page-quantum rounding still departs from the LP when a call has fewer pages than columns. The open 64 KiB reduce-scatter regression (1.4-2.4 ms a call) is not explained by this plan.
- **The LP prices no combine time.** A pool slower than its links at a relay invalidates T* there.
- **Masked calls** still use spanning columns.
- **Cable bandwidths are the ports' stated rates.** Their ratios hold here because every cable is identical.
- **Scaling.** The LP must be normalised: unscaled, HiGHS reported 2.571 ms for the all-gather, whose optimum is 2.057.
- **Files outside mesh.** Steps 9 and 10 touch mesh_rank.m (another group's file; minimal hunks) and example configurations.
- **Line numbers will drift.** They are as of ad4a8bdd.
