#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent);
/* M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, section 5.2, step 3: "Sort
   instructions into a global topological order with respect to their dependencies with a heap using the priority to
   order instructions."  priority(v) = -v. */
int mesh_graph_order(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                     const void *context, uint32_t *indegree, uint32_t *ordinal);
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer);
void mesh_graph_rows(uint32_t keys, size_t items, uint32_t (*key)(const void *context, size_t k), const void *context,
                     size_t *first, size_t *order);
uint32_t mesh_graph_keyed_find(const uint32_t *head, const uint32_t *next, uint32_t key,
                               int (*equal)(const void *context, uint32_t k), const void *context);
void mesh_graph_keyed_add(uint32_t *head, uint32_t *next, uint32_t key, uint32_t k);
double mesh_graph_exp(double x);
double mesh_graph_log(double x);
uint32_t mesh_graph_flow(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share);
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index);
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count);
/* R. L. Graham, "Bounds for certain multiprocessing anomalies", Bell System Technical Journal 45 (1966). */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried);
/* E. W. Dijkstra, A note on two problems in connexion with graphs, Numerische Mathematik 1, 1959.
   lags[s * targets + x] = min over walks w of one or more edges from source[s] to target[x] of sum over k in w of
   lag(context, k), UINT32_MAX where there is none. */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint32_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint32_t *lags);
/* R. P. Dilworth, A decomposition theorem for partially ordered sets, Annals of Mathematics 51(1), 1950.
   D. R. Fulkerson, Note on Dilworth's decomposition theorem for partially ordered sets, Proc. AMS 7(4), 1956.
   J. E. Hopcroft and R. M. Karp, An n^5/2 algorithm for maximum matchings in bipartite graphs, SIAM J. Comput. 2(4),
   1973.  E. A. Dinic, Algorithm for solution of a problem of maximum flow in a network with power estimation, Soviet
   Math. Doklady 11, 1970.
   *chains = sum over i of count[i] - |M|, M a maximum matching; chain[o_i + k], o_i = sum over j < i of count[j]. */
int mesh_graph_chains(uint32_t items, const uint64_t *count, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, uint64_t *chain, uint64_t *chains);
/* C. Mukerji, Register allocation using cyclic interval graphs, M.Sc. thesis, McGill University, 1994:
   - Definition 2.1.1: "A time t is covered by ... an interval I1' : [t1, t1'] if (t1 ≤ t ≤ t1'), or by a cyclic
     interval I2 : ([t1, t1'), [t2, t2'])"; §3.1: "Cyclic intervals wrap around from one iteration to the next".
     Interval i = [start[i], end[i]] on 0 .. length - 1, cyclic iff end[i] < start[i].
   - Algorithm 3.2.1's input, p. 43: "The number of registers, reg_num, needed by each interval of the cyclic interval
     graph."  reg_num(i) = need[i]; color[o_i + k], k < need[i], o_i = sum over j < i of need[j].
   - §3.2.6, Main Fatcover, Establish All Covers Of C_i and Choose A Cover For C_i; Appendix A, Algorithm A.0.1. */
int mesh_graph_fat_cover(uint64_t length, uint32_t intervals, const uint64_t *start, const uint64_t *end,
                         const uint64_t *need, uint64_t *color, uint64_t *colors);
/* C.-Y. Chen, Scheduling coflows with precedence constraints for minimizing the total weighted completion time in
   identical parallel networks, arXiv 2205.02474, 2022, (1b) to (1e), (14) and (14a).  L. A. Hall, A. S. Schulz,
   D. B. Shmoys and J. Wein, Scheduling to minimize average completion time, Math. Oper. Res. 22(3), 1997, (3) and (5).
   M. Queyranne, Structure of a simple scheduling polyhedron, Math. Programming 58, 1993.  Q. Huangfu and J. A. J.
   Hall, Parallelizing the dual revised simplex method, Math. Program. Comput. 10(1), 2018.  H. Isermann, Linear
   lexicographic optimization, OR Spektrum 4, 1982.
   d_j = length[j].  First minimize C_max; then, C_max fixed at that optimum, minimize sum over j of C_j:
     C_max >= C_j for every j;
     C_j >= d_j;
     C_j >= C_i + d_j for every edge i = from[e] -> j = to[e];
     sum over j in S of d_j C_j >= (d(S)^2 + d^2(S)) / 2 for every resource r and S a subset of {j : resource[j] = r}.
   completion[j] = C_j. */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion);
/* M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, section 5.2, step 3: "Sort
   instructions into a global topological order with respect to their dependencies with a heap using the priority to
   order instructions." */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP systems, IEEE Trans.
   Signal Processing 45(6), 1997 (BSL).  delay[e] = UINT64_MAX is BSL's "∞"; time[v] is BSL §4.1's "execution time
   t(v)". */
/* BSL §9: "takes a synchronization graph Z as input, and applies the Bellman-Ford algorithm discussed in pp. 94-97 of
   [15] to return the cycle mean of the critical cycle in Z; if one or more cycles exist that have zero path delay,
   then BellmanFord returns ∞".  [15]: E. L. Lawler, Combinatorial Optimization: Networks and Matroids, 1976. */
int mesh_graph_cycle_mean(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          const uint64_t *delay, double *mean);
/* BSL Fig. 8, Function MinDelay(X, e, λ, B): "Perform a binary search in the range [0, 1, …, B] to find the minimum
   value of r ∈ {0, 1, …, B} such that BellmanFord(X[e → r]) returns a value less than or equal to λ.  Return this
   minimum value of r". */
int mesh_graph_min_delay(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                         const uint64_t *delay, size_t e, double lambda, uint64_t bound, uint64_t *least);
/* BSL §9, Fig. 6, Function Convert-to-SC-graph.  BSL §3: "An SCC V′ of a DFG (V, E) is a source SCC if ∀e ∈ E,
   (snk(e) ∈ V′) ⇒ (src(e) ∈ V′); V′ is a sink SCC if (src(e) ∈ V′) ⇒ (snk(e) ∈ V′)".  BSL §9: "(e_0, e_1, …,
   e_m – 1) ≡ (e_s, a_1, a_2, …, a_k, b_l, b_l – 1, …, b_1), where e_s is the sink-source edge".  Theorem 4's proof:
   "(n src + n snk – 1)" edges, at most 2 |V| - 1. */
int mesh_graph_convert_sc(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *added, uint32_t *added_from, uint32_t *added_to);
/* BSL §9, Fig. 8, Function DetermineDelays: "X_0 = Ĝs[e_0 → ∞, …, e_m – 1 → ∞]; λ_max = BellmanFord(X_0); d_ub =
   ⌈Σ_{x∈V} t(x) / λ_max⌉; For i = 0, 1, …, m – 1: δ_i = MinDelay(X_i, e_i, λ_max, d_ub); X_i + 1 = X_i[e_i → δ_i]".
   added_delay[i] = δ_i. */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, uint64_t *added_delay);
/* MPI 4.1 §6.1.11 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node107.htm): "A datatype may specify
   overlapping entries. The use of such a datatype in any communication in association with a buffer updated by the
   operation is erroneous."
   1 iff, sorted by (key, start), some k has key_k = key_{k-1} and start_k < end_{k-1}. */
struct mesh_graph_interval { uint64_t key, start, end; };
int mesh_graph_intersect(size_t count, struct mesh_graph_interval *interval);
void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context);

#ifdef __cplusplus
}
#endif
#endif
