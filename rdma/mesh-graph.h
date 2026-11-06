/* design/rubrics.md:422 (C9): the graph algorithms, one copy each, for every consumer of weights or samples on a graph.
   A function here knows vertices, edges and the operands its consumer passes (weights, visits, a next hop); it knows
   nothing of what they stand for, no collective, tensor or transfer.  Python's are rdma/mesh_graph.py's. */
/* Source: pre-return-20261103 rdma/mesh-graph.h, the declarations of the functions mesh-graph.c brings, verbatim;
   mesh_graph_list_schedule is new (design/page-queue-change-request.md, "The rule both ends compute");
   mesh_graph_left_edge is new (design/disjoint-pages-change-request.md, Appendix A, A3, "The assignment";
   design/disjoint-pages-coordination.md, "The allocator"); mesh_graph_lags is new (design/disjoint-pages-change-request.md, Appendix C, C3.3;
   design/compilation-coordination.md, U1; design/disjoint-pages-revision-coordination.md, "The allocator");
   mesh_graph_bottom_levels and mesh_graph_contention_schedule are new, and mesh_graph_reach gains `parent`
   (design/plan-to-bridge-change-request.md, R5; design/compilation-coordination.md, "The graph module" and T3 to T5,
   Q3); mesh_graph_heap_order is new (design/plan-to-bridge-change-request.md, R5, "The order of each cable's list, by
   the reference algorithm"; design/compilation-coordination.md, "The documents' revisions of edd4b583", "The graph
   module" and GO3), its heap mesh_graph_lags' binary heap, one copy.
   2026-11-04, owner C2 (design/compilation-coordination.md, U1): mesh_graph_lags' source cited to C3.3 and U1, where it
   cited design/disjoint-pages-revision-coordination.md, D5.  Why: D5 appears in neither Lam, Rau nor C3.3's text
   (design/aot-compilation-change-request.md, appendix item 1); the least lags decide C3.3's tests at every larger
   distance by U1's reason, its consumer's (every walk into a write passes its range's production; the programs'
   dependences order each node's events of t before those of t + 1).  What must stay: the function as it is.
   2026-11-05 (design/compilation-coordination.md, "The revisions of 46ae8fdf", LP; design/plan-to-bridge-change-
   request.md R5 at mesh 0e56d8da): mesh_graph_bottom_levels and mesh_graph_contention_schedule are removed, and
   mesh_graph_completion_times is new.  Why: R5's priority is "C̄_j, its completion time in the program's optimal
   solution, the least first", and "It replaces MSCCLang's depth and reverse depth, and edge scheduling's start times,
   which are no longer computed for the order"; R5's Code: "rdma/mesh-graph.{h,c}: the program's rows, the
   cutting-plane loop with Queyranne's separation, and the calls to HiGHS; the heap order by C̄ (`mesh_graph_heap_order`,
   as now)".  What must stay: mesh_graph_heap_order as it is, the heap one copy */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* The vertices `from` reaches over `edges` undirected edges, edge e joining `end(context, e, 0)` and
   `end(context, e, 1)` (each below `vertices`), breadth first: their count; `order` (room for `vertices`) the vertices
   in the order reached, `reached` (room for `vertices`) 1 for each and 0 for every other.  `parent` (room for
   `vertices`; NULL: none): each reached vertex's breadth-first parent, the vertex whose edge first reached it (`from`
   its own; a vertex not reached UINT32_MAX).  Callers pass NULL but mesh-records.c's island tree
   (design/compilation-coordination.md, Q3). */
uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent);
/* Directed edges in compressed rows: vertex v's leave as edges [first[v], first[v + 1]), edge k entering
   `head(context, k)`. */
/* A topological order of such edges' `vertices` vertices: ordinal[v] v's place, each place taken by the
   lowest-numbered vertex whose every edge in leaves a placed one.  `indegree` holds `vertices` words.  0, or EDEADLK
   where a cycle leaves vertices unplaced (their ordinal UINT32_MAX). */
int mesh_graph_order(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                     const void *context, uint32_t *indegree, uint32_t *ordinal);
/* Each marked vertex's depth over the edges into it (vertex v's row [first[v], first[v + 1]) of head[]: the vertices
   whose work it follows): 0 where its row reaches no marked vertex, else one past the greatest depth among the marked
   ones it reaches, taken over a topological order (mesh_graph_order).  0; EDEADLK where the rows hold a cycle; ENOMEM */
int mesh_graph_depth(uint32_t vertices, const size_t *first, const uint32_t *head, const unsigned char *marked, uint32_t *depth);
/* Chu-Liu/Edmonds: the minimum spanning arborescence of nodes [0, N) rooted at `root` in which each other node v
   takes one edge e with to[e] == v (its parent from[e]); chosen[v] that edge, the root's -1; 0, or -1 where a node has
   none.  Each node's lightest edge; a cycle among them contracted, the edges into it weighing their excess over the
   edge they would replace, the contracted graph's arborescence expanded (the edge into a cycle replaces the cycle's
   edge into the same node). */
int mesh_graph_arborescence(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                            const double *weight, int32_t *chosen);
/* The least-weight spanning arborescence of nodes [0, N) rooted at `root` in which every node lies within `depth`
   edges of the root (a hop-bounded arborescence; Chu-Liu/Edmonds' has no bound): chosen[v] the edge into v, the
   root's -1; 0, or -1 where no such arborescence spans.  Exact, by depth-first search over each node's edges in, a
   branch cut where its weight with each node left's lightest edge in reaches the best found or a node's walk to the
   root passes `depth`: its cost the product of the nodes' in-degrees at worst (the problem is NP-hard for depth 2
   and beyond), a few hundred leaves on a collective of five. */
int mesh_graph_arborescence_within(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                                   const double *weight, uint32_t depth, int32_t *chosen);
/* The walk up a parent table from `from` (parent[v]: v's next vertex up): `visit(consumer, at, step)` for each vertex
   it meets, `from` at step 0, until one that is `root` (UINT32_MAX: none) or whose parent names none
   (parent[at] >= vertices), visited last; a vertex its own parent (outside a tree) climbs in place.  The steps to that vertex; UINT32_MAX where `vertices` steps meet none (a
   cycle), steps 0 to `vertices` visited.  `visit` NULL: none. */
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer);
/* The length of the cycle a climb from `from` returns to `from` on (the first step past 0 it meets `from` again);
   0 where it does not within `vertices` steps. */
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from);
/* Records grouped by a key into compressed rows (a routing's trees by their demand, its paths by their scope): record
   k's key is key(context, k) < keys; key v's records are order[first[v]] to order[first[v + 1] - 1], in record order.
   first holds keys + 1 entries and order items: one count, one sum and one scatter. */
void mesh_graph_rows(uint32_t keys, size_t items, uint32_t (*key)(const void *context, size_t k), const void *context,
                     size_t *first, size_t *order);
/* Items chained by a key, latest first, as a solver finds them (its distinct trees of one demand or of one root):
   head[key] that key's latest item (UINT32_MAX: none) and next[k] the one before item k. mesh_graph_keyed_find gives
   the key's item for which equal(context, k) holds, or UINT32_MAX, and reads that key's items alone;
   mesh_graph_keyed_add chains item k to its key. */
uint32_t mesh_graph_keyed_find(const uint32_t *head, const uint32_t *next, uint32_t key,
                               int (*equal)(const void *context, uint32_t k), const void *context);
void mesh_graph_keyed_add(uint32_t *head, uint32_t *next, uint32_t key, uint32_t k);
/* exp and log of IEEE-exact operations alone (+, -, x, / correctly rounded; frexp, ldexp, floor exact), so that every
   node computes the same trees and the same shares from them, whatever its libm */
double mesh_graph_exp(double x);
double mesh_graph_log(double x);
/* A concurrent flow of demands over an undirected graph by multiplicative weights (a planner's routing solve:
   mesh-plan.h mesh_solve). Edge e joins from[e] and to[e], capacity[e] each way. Demand d's tree is rooted at root[d]
   and reaches each vertex v for which leaf(context, d, v) holds; up[d] bytes cross it toward the root and down[d] from
   it. Each of `rounds` rounds routes every demand along its least-length tree, a direction's length
   exp(eps (load / capacity - most) / width) / capacity weighed by the bytes that cross it, and adds a round's share of
   those bytes to the loads. Each demand's distinct trees are kept, in the order found: tree k's demand demand[k], its
   parents parent[k * vertices ..] (the root its own, a vertex off the tree UINT32_MAX), its share share[k] (the
   rounds that chose it over all rounds). Their count; UINT32_MAX where eps is not positive, a root is no vertex, a leaf
   has no route, or more than room trees are found. */
uint32_t mesh_graph_flow(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share);
/* Spanning trees packed by Garg and Koenemann's rounds (a communicator's own forms: mesh-plan.h mesh_trees_pack). Edge e
   makes vertex to[e] a child of from[e], its route crossing directed hops crossing[through[e]] to
   crossing[through[e + 1] - 1], hop k of capacity[k]. Every root r for which rooted(context, r) holds is given a demand
   of one, packed along least-weight arborescences (mesh_graph_arborescence) until the hops' lengths reach one. Each
   root's `most` heaviest distinct trees are kept, root by root: root[k], parents parent[k * vertices ..] and log
   weight log_weight[k], the log of its share of its root's kept flow. Their count; 0 where no arborescence spans.
   `shallow`: each root's arborescences are those within its eccentricity's depth alone (the most edges its
   breadth-first layering puts between it and a vertex; mesh_graph_arborescence_within), so each tree is as shallow as
   a tree from that root can be: where a hop costs a fixed time far past its bytes' (a small collective), a tree's
   time is its depth's hops. */
uint32_t mesh_graph_pack(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const uint32_t *through,
                         const uint32_t *crossing, uint32_t hops, const double *capacity,
                         int (*rooted)(const void *context, uint32_t r), const void *context, double eps, uint32_t most,
                         int shallow, uint32_t *root, uint32_t *parent, double *log_weight);
/* Trees' loads on their directed edges: each tree k (root root[k], parents parent[k * vertices ..]) of weight weight[k]
   adds weight[k] to load[u * vertices + v] for each of its edges u -> v (u v's parent); `load` is the caller's,
   vertices x vertices, added to. */
void mesh_graph_tree_loads(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                           const double *weight, double *load);
/* [first, first + length) cut at the multiples of `cell` (a grid over the whole operand): start[i], count[i] and the
   cell's index index[i] of each run; their number. A run inside another's range is cut at the same edges, so each of
   its pieces lies within one of the other's (a consumer that gives each cell, by its index, to one of several parallel
   machines gives an inner run's piece the machine of the outer piece holding it, and two parties cutting the same
   operands give it alike). */
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index);
/* A tree's depth (its root root, parents parent[..], a vertex off it UINT32_MAX): the most edges from the root to a
   vertex on it, mesh_graph_depth's over each vertex's parent; UINT32_MAX where its parents hold a cycle */
uint32_t mesh_graph_tree_depth(uint32_t vertices, uint32_t root, const uint32_t *parent);
/* The rate every root gets at once where each root's flow is cut over its trees by their weights (weights a root's
   summing to one): the least, over directed edges u -> v, of capacity[u * vertices + v] over the trees' load there
   (mesh_graph_tree_loads), in capacity's units; 0 where an edge a tree crosses has no capacity. */
double mesh_graph_trees_rate(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                             const double *weight, const double *capacity);
/* `units` equal units over `parts` parallel routes by earliest finish: part p takes its first unit latency[p] after
   the start and then one each 1 / rate[p] (rate in units a time); unit k goes where it would finish soonest, so a
   unit takes the quickest route and many spread over every route with rate, a slower route taking fewer of the first
   (makespan of equal jobs on uniform machines with release times, which earliest finish solves exactly). count[p]
   its units; their completion time. A part of no rate takes none. */
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count);
/* Units in list order over `machines` parallel machines by earliest finish (Graham's list scheduling: R. L. Graham,
   "Bounds for certain multiprocessing anomalies", Bell System Technical Journal 45 (1966); each machine of its own
   rate, so equal rates give Graham's least-loaded machine): unit k, bytes[k] long, goes to the machine l whose bytes
   with unit k's added, over rate[l] (bytes a time), end soonest, a tie to the lower l, so parallel machines of unequal
   rate end together (the machines in the caller's order; its arithmetic is +, / and < alone, correctly rounded, so two
   parties computing from the same operands assign one list alike). machine[k] its machine (UINT32_MAX where no machine
   has rate), carried[l] (room for `machines`) the bytes each takes; their completion time, INFINITY where a unit has no
   machine. A machine of no rate takes none. */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried);
/* 2026-11-04, sub-assignment 2: mesh_graph_happens_before (b399746f), a bit table of the paths over a template
   unrolled across `iterations` executions, is removed with its helpers in mesh-graph.c.  Why: C3.3's unroll factor u,
   each value's L and every interference test come from least lags over the template alone (mesh_graph_lags;
   design/disjoint-pages-change-request.md, C3.3; design/compilation-coordination.md, U1, where this note cited
   design/disjoint-pages-revision-coordination.md, D5, 2026-11-04), never from a table over the run's executions,
   which made a plan's compile time grow with the run's length (C1); its last caller, mesh-collective.c's 2026-11-03
   allocator, went in baede218.  What must stay: nothing in a plan's graph computation
   depends on the run's length; a lag-0 cycle is found by mesh_graph_order over the lag-0 edges alone (EDEADLK), as
   this function found one and mesh-collective.c now does. */
/* The least total lag over walks of one or more edges from source[s] to target[x] (s < sources, x < targets), in the
   directed graph whose vertex v's edges are [first[v], first[v + 1]), edge k entering head(context, k) and carrying
   lag(context, k) >= 0 (E. W. Dijkstra, A note on two problems in connexion with graphs, Numerische Mathematik 1,
   1959): lags[s * targets + x], UINT32_MAX where no walk joins them.  A source that is its own target is given its
   least cycle.  0; ENOMEM */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint32_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint32_t *lags);
/* 2026-11-05 (design/compilation-coordination.md, "The documents' corrections of 2026-11-05", point 6: "Greedy by size
   goes, with `mesh_graph_left_edge`'s colouring"; "The revisions of 4e78f6f4"): mesh_graph_left_edge is removed, the
   left-edge colouring of items that each needed a run of colours, greedy by size (T. Hashimoto and J. Stevens, DAC
   1971; Y. Pisarchyk and J. Lee, arXiv 2001.03288, 2020).  Why: R2, corrected: "No value is placed as one run.  So
   greedy by size ... and the contiguous-block problem it approximates ... have no use here.  The chains replace
   C3.3's colouring."  What must stay: placement by an operand of the caller's, here mesh_graph_chains */
/* The fewest chains of a partial order (design/plan-to-bridge-change-request.md R2, corrected 2026-11-05: "By
   Dilworth's theorem that number equals the largest set of items no two of which are ordered (R. P. Dilworth, A
   decomposition theorem for partially ordered sets, Annals of Mathematics 51(1), 1950).  It is found by maximum
   bipartite matching (D. R. Fulkerson, Note on Dilworth's decomposition theorem for partially ordered sets, Proc. AMS
   7(4), 1956), exactly and in polynomial time"; design/disjoint-pages-change-request.md, A3 and C3.3, corrected).
   The caller's items 0 .. items - 1, item i of count[i] >= 1 elements, which are mutually unordered and ordered as
   item i is: before(context, a, b) != 0 where a precedes b, a strict partial order the caller gives (irreflexive and
   transitive).  Fulkerson's bipartite graph has each element on both sides and an edge from x to y wherever x
   precedes y; a maximum matching of it, by Hopcroft and Karp's phases (J. E. Hopcroft and R. M. Karp, SIAM J. Comput.
   2(4), 1973, as rdma/mesh_graph.py `matching`), with an item's elements as one vertex of capacity count[i] on each
   side, is a maximum flow of that network (Hopcroft and Karp's phases on a network of capacities are E. A. Dinic's
   blocking flows, Soviet Math. Doklady 11, 1970), and the fewest chains are the elements less its value.  Out: *chains
   their number, and chain[o_i + k] the chain of item i's element k, k < count[i], o_i the sum of count[j] over j < i:
   each chain a matched path x_1 -> x_2 -> ... of elements, so its elements are mutually ordered; the chains numbered
   from 0 in the order of their first elements, items in the caller's order and an item's elements in turn, so an
   item that precedes and follows none has chains of its own, consecutive.  Every party computing from the same
   operands numbers the chains alike.  0; ENOMEM.
   2026-11-05 (design/compilation-coordination.md, "Review of 4e78f6f4 against the requirement lines", G1; "The
   revisions of 55b9cbad"): the items' linear extension is mesh_graph_order's, so EDEADLK where `before`'s pairs hold
   a cycle, which a strict partial order cannot (nothing written).  What must stay: the chains' numbering */
int mesh_graph_chains(uint32_t items, const uint64_t *count, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, uint64_t *chain, uint64_t *chains);
/* design/plan-to-bridge-change-request.md R5 (mesh 0e56d8da), "How: the program" and "How: solving it": the completion
   times C̄ of one linear program over a directed graph, vertex j of length length[j] >= 0 on resource resource[j] (<
   `resources`), edge e running from[e] -> to[e]: variables C_j and C_max; minimize C_max with C_max >= C_j for every j
   (C.-Y. Chen, arXiv 2205.02474, 2022, (14) and (14a)); C_j >= d_j (Chen's (1b), every release time zero); C_j >= C_i +
   d_j for every edge i -> j (Chen's (1c), "the precedence constraint"; L. A. Hall, A. S. Schulz, D. B. Shmoys and J.
   Wein, Math. Oper. Res. 22(3) 1997, (3)); and for each resource and each set S of its vertices, sum over j in S of d_j
   C_j >= (d(S)^2 + d^2(S)) / 2 (Hall et al.'s (5), "introduced by Queyranne (1993)"; Chen's (1d) and (1e) with m = 1).
   Solved by cutting planes: each linear program by HiGHS's dual revised simplex (Q. Huangfu and J. A. J. Hall, Math.
   Program. Comput. 10(1), 2018), each resource's violated rows found by Queyranne's separation (M. Queyranne, Math.
   Programming 58, 1993) and added, until none is violated (design/compilation-coordination.md, "The revisions of
   46ae8fdf", LP2 to LP6 and LP10); then, with C_max held at that optimum, minimize the sum of the C_j by the same
   rounds (R5 at mesh f3f7cf46: "one lexicographic linear program (H. Isermann, Linear lexicographic optimization, OR
   Spektrum 4, 1982)"; "The revisions of f3f7cf46", LX).  completion[j] out: C̄_j of the second objective's last
   solution.  Every party solving the same
   operands with the same library computes the same C̄.  0; EINVAL where a length is negative or not finite, a resource or an edge
   names none, an edge runs from a vertex to itself, or the program has no solution (a cycle of positive length);
   EDOM where HiGHS reports an error or a status other than optimal or infeasible; ENOMEM, also where the program's
   rows pass HiGHS's 32-bit indices */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion);
/* M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, section 5.2, step 3: "Sort
   instructions into a global topological order with respect to their dependencies with a heap using the priority to
   order instructions."  Vertex v's priority priority[v]; edge e runs from[e] -> to[e] (an edge given twice counts
   twice and is removed twice, so it changes nothing).  A vertex enters the heap once every edge into it leaves a
   placed vertex; the heap's first is placed next, the greatest priority first and a tie to the lower vertex
   (design/compilation-coordination.md, "The documents' revisions of edd4b583", GO3; R5's "the least C̄_j first" is
   its consumer's priority -C̄: "The revisions of 46ae8fdf", LP7).  place[v] out, v's place in the order.  0; EINVAL
   where an edge names no vertex or the edges hold a cycle (a vertex left unplaced); ENOMEM */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place);
/* 2026-11-05 (design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the
   self-timed buffer bound": "The strongly connected components, Convert-to-SC-graph, Bellman–Ford's cycle mean and
   MinDelay's binary search are functions of rdma/mesh_graph.py.  They take the caller's vertices, edges, delays and
   times as operands"; design/compilation-coordination.md, "The revisions of 3f106747", BS-1: here, the graph module's C
   half, since their one caller is mesh_plan, which holds the routes and u they are computed from).  S. S.
   Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP systems, IEEE Trans. Signal
   Processing 45(6), 1997 (here BSL), over a directed graph of `vertices` vertices whose edge e runs from[e] -> to[e]
   with delay[e] (UINT64_MAX: BSL's "∞", the edge absent from every cycle) and whose vertex v takes time[v] >= 0 (BSL
   §4.1, "execution time t(v)").  What must stay: integer and IEEE-exact operations alone (+, -, x, /, <), so every
   party computes the same components, means and delays from the same operands, as the module's other functions do */
/* The strongly connected components (BSL §3: "A strongly connected component (SCC) of (V, E) is a strongly connected
   subset V′ ⊆ V such that no strongly connected subset of V properly contains V′"), by R. E. Tarjan's depth-first
   search (Depth-first search and linear graph algorithms, SIAM J. Comput. 1(2), 1972), every edge counted whatever its
   delay: component[v] v's, *count of them, numbered in the order Tarjan's search completes them.  0; EINVAL where an
   edge names no vertex; ENOMEM */
int mesh_graph_components(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *component, uint32_t *count);
/* BSL's BellmanFord (§9: "takes a synchronization graph Z as input, and applies the Bellman-Ford algorithm discussed in
   pp. 94-97 of [15] to return the cycle mean of the critical cycle in Z; if one or more cycles exist that have zero
   path delay, then BellmanFord returns ∞"): *mean the maximum over cycles C of the sum of time[v] over C's vertices
   over Delay(C) (BSL (4), Definition 2), INFINITY where a cycle of the edges not absent has zero total delay (BSL
   Lemma 1: deadlock), 0 where they hold no cycle.  [15] is E. L. Lawler, Combinatorial Optimization: Networks and
   Matroids, 1976: a binary search on λ, each step Bellman and Ford's test for a cycle of positive length under the
   lengths time[from[e]] - λ delay[e]; *mean the search's upper end, at which no cycle is positive.  0; EINVAL; ENOMEM */
int mesh_graph_cycle_mean(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          const uint64_t *delay, double *mean);
/* BSL Fig. 8, Function MinDelay(X, e, λ, B): "Perform a binary search in the range [0, 1, …, B] to find the minimum
   value of r ∈ {0, 1, …, B} such that BellmanFord(X[e → r]) returns a value less than or equal to λ.  Return this
   minimum value of r", X the operands' graph, e < edges, its delay[e] replaced by r and every other edge's kept; *least
   r.  Each step's test is mesh_graph_cycle_mean's at λ: no cycle of zero total delay, and none of positive length.  0;
   ERANGE where r = `bound` fails (BSL's input: "Assuming X[e → B] has estimated throughput no less than λ^-1"); EINVAL;
   ENOMEM */
int mesh_graph_min_delay(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                         const uint64_t *delay, size_t e, double lambda, uint64_t bound, uint64_t *least);
/* BSL §9, Fig. 6, Function Convert-to-SC-graph: the edges, each of delay 0 (BSL's d0), that chain the source SCCs
   C_1 .. C_m (each the vertex of its SCC of least time, a tie to the lower vertex: "Select a vertex v_i ∈ C_i that
   minimizes t(*) over C_i"), chain the sink SCCs D_1 .. D_n alike, and join the last sink SCC's to the first source
   SCC's (step 6, "the sink-source edge"); each ordering of SCCs ascending by its least vertex ("Generate an
   ordering").  BSL §3: "An SCC V′ of a DFG (V, E) is a source SCC if ∀e ∈ E, (snk(e) ∈ V′) ⇒ (src(e) ∈ V′); V′ is a
   sink SCC if (src(e) ∈ V′) ⇒ (snk(e) ∈ V′)".  Out, in the order DetermineDelays takes them (BSL §9: "(e_0, e_1, …,
   e_m – 1) ≡ (e_s, a_1, a_2, …, a_k, b_l, b_l – 1, …, b_1), where e_s is the sink-source edge"): added_from[i] ->
   added_to[i], *added of them (room for 2 x `vertices`: "(n src + n snk – 1)", Theorem 4's proof, reaches 2 |V| - 1
   where SCCs of one vertex are both source and sink; 2026-11-05, "The revisions of 3f106747", G-3, where it said
   `vertices`); 0 added where the graph is strongly connected.  0; EINVAL; ENOMEM */
int mesh_graph_convert_sc(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *added, uint32_t *added_from, uint32_t *added_to);
/* BSL §9, Fig. 8, Function DetermineDelays, over the operands' graph Gs and the `added` edges mesh_graph_convert_sc
   gave, in its order: "X_0 = Ĝs[e_0 → ∞, …, e_m – 1 → ∞]; λ_max = BellmanFord(X_0); d_ub = ⌈Σ_{x∈V} t(x) / λ_max⌉;
   For i = 0, 1, …, m – 1: δ_i = MinDelay(X_i, e_i, λ_max, d_ub); X_i + 1 = X_i[e_i → δ_i]", d_ub 1 where λ_max is 0
   (every cycle's time 0, so any delay of at least 1 keeps a new cycle's mean at 0).  added_delay[i] out, δ_i.  0;
   EDEADLK where Gs holds a cycle of zero total delay (BSL Lemma 1); ERANGE (mesh_graph_min_delay's); EINVAL; ENOMEM */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, uint64_t *added_delay);
/* The sweep test for interval intersection (design/problems-of-14b9a757-change-request.md, item 9: "the destination
   intervals of each node and execution are sorted by start.  Any interval that begins before its predecessor ends
   makes the plan EINVAL.  That is the sweep test for interval intersection"): interval[0 .. count), each [start, end)
   with start < end, sorted in place by key, then start; an interval is compared with those of its key alone.  1 where
   an interval begins before its predecessor of the same key ends, else 0 */
struct mesh_graph_interval { uint64_t key, start, end; };
int mesh_graph_intersect(size_t count, struct mesh_graph_interval *interval);

/* A vertex's adjacency row, row[0 .. count) (mesh_graph_rows), visited in order: visit(context, row[j]) */
void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context);

#ifdef __cplusplus
}
#endif
#endif
