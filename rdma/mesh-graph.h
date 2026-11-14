/* design/rubrics.md:422 (C9) */
#ifndef MESH_GRAPH_H
#define MESH_GRAPH_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

uint32_t mesh_graph_reach(uint32_t vertices, uint32_t edges, uint32_t (*end)(const void *context, uint32_t e, uint32_t side),
                          const void *context, uint32_t from, uint32_t *order, unsigned char *reached, uint32_t *parent);
int mesh_graph_order(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                     const void *context, uint32_t *indegree, uint32_t *ordinal);
/* T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, Introduction to Algorithms, 3rd ed., MIT Press, 2009,
   §22.3 and §22.4, Lemma 22.11: depth-first search over rows (first, head) from `root`, each reachable vertex visited
   once in preorder with its tree parent (root's UINT32_MAX) after the search; cross and forward edges allowed (a callee
   of two callers is visited once), EDEADLK on a back edge (a cycle) */
int mesh_graph_preorder(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                        const void *context, uint32_t root,
                        int (*visit)(void *consumer, uint32_t vertex, uint32_t parent), void *consumer);
/* the depth-first unfolding of rows (first, head) from `root`: each path from root is one instance, visited in preorder
   with its vertex, the row entry k it was reached by (SIZE_MAX for the root) and its parent instance (UINT32_MAX for the
   root); instances are numbered in visit order from 0; a vertex reached by several paths is visited once per path;
   EDEADLK on a back edge (a cycle), EOVERFLOW where an instance would need number UINT32_MAX (the parent sentinel),
   the visit's nonzero return ends the search with it */
int mesh_graph_unfold(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                      const void *context, uint32_t root,
                      int (*visit)(void *consumer, uint32_t vertex, size_t k, uint32_t parent), void *consumer);
/* one owned search of rows (first, head): count visited vertices, preorder[0 .. count) in discovery (depth-first) or
   visit (breadth-first) order, postorder[0 .. count) in finishing order (depth-first; NULL breadth-first), parent[v]
   and entry[v] the tree parent and row entry that reached v (UINT32_MAX and SIZE_MAX for a root or an unvisited
   vertex), place[v] v's index in preorder (UINT32_MAX unvisited); freed by mesh_graph_search_free */
struct mesh_graph_search { uint32_t vertices, count; uint32_t *preorder, *postorder, *parent, *place; size_t *entry; };
/* T. H. Cormen et al., 3rd ed., §22.3 and §22.4, Lemma 22.11: depth-first search over rows (first, head) from
   root[0 .. roots) in that order, each row's entries in order (in reverse order where `reverse`), each reachable vertex
   visited once; EDEADLK on a back edge (a cycle), EINVAL on a vertex out of range */
int mesh_graph_search(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                      const void *context, uint32_t roots, const uint32_t *root, int reverse,
                      struct mesh_graph_search **made);
/* T. H. Cormen et al., 3rd ed., §22.2: breadth-first search over rows (first, head) from start[0 .. starts), each start
   and then each row entry in order, a vertex entering once; EINVAL on a vertex out of range */
int mesh_graph_breadth(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                       const void *context, uint32_t starts, const uint32_t *start, struct mesh_graph_search **made);
void mesh_graph_search_free(struct mesh_graph_search *made);
/* rows held as an array: ((const uint32_t *)context)[k] */
uint32_t mesh_graph_row_head(const void *context, size_t k);
/* rows of a retained relation (context its struct mesh_graph_relation_data, first its row_first): edge[row_edge[k]].to */
uint32_t mesh_graph_relation_head(const void *context, size_t k);
/* A. Goralčíková and V. Koubek, LNCS 74, 1979; K. Simon, TCS 58, 1988: over rows (first, head) whose heads all precede
   their row (`following` 0) or all follow it (`following` 1) in vertex order, bit u of reached[v * words + u / 64] set
   where a path of rows leads from v to u, words = (vertices + 63) / 64; EINVAL where a head is on the other side of its
   row; freed by mesh_graph_closure_free */
struct mesh_graph_closure { uint32_t vertices; size_t words; uint64_t *reached; };
int mesh_graph_closure(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                       const void *context, int following, struct mesh_graph_closure **made);
void mesh_graph_closure_free(struct mesh_graph_closure *made);
int mesh_graph_depth(uint32_t vertices, const size_t *first, const uint32_t *head, const unsigned char *marked, uint32_t *depth);
/* Y. J. Chu and T. H. Liu, Scientia Sinica 14, 1965; J. Edmonds, J. Res. Nat. Bur. Standards 71B, 1967 */
int mesh_graph_arborescence(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                            const double *weight, int32_t *chosen);
int mesh_graph_arborescence_within(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                                   const double *weight, uint32_t depth, int32_t *chosen);
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer);
void mesh_graph_rows(uint32_t keys, size_t items, uint32_t (*key)(const void *context, size_t k), const void *context,
                     size_t *first, size_t *order);
uint32_t mesh_graph_keyed_find(const uint32_t *head, const uint32_t *next, uint32_t key,
                               int (*equal)(const void *context, uint32_t k), const void *context);
void mesh_graph_keyed_add(uint32_t *head, uint32_t *next, uint32_t key, uint32_t k);
double mesh_graph_exp(double x);
double mesh_graph_log(double x);
/* design/joint-compilation.md §3, row 3 */
uint32_t mesh_graph_flow(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share);
/* N. Garg and J. Könemann, FOCS 1998 */
uint32_t mesh_graph_pack(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const uint32_t *through,
                         const uint32_t *crossing, uint32_t hops, const double *capacity,
                         int (*rooted)(const void *context, uint32_t r), const void *context, double eps, uint32_t most,
                         int shallow, uint32_t *root, uint32_t *parent, double *log_weight);
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index);
uint32_t mesh_graph_tree_depth(uint32_t vertices, uint32_t root, const uint32_t *parent);
/* design/joint-compilation.md §3, row 4 */
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count);
/* design/joint-compilation.md §3, row 10 */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried);
/* ../../metal-microbench/docs/provider.md#composed-resource-costs */
enum { MESH_GRAPH_SCALAR, MESH_GRAPH_MATRIX, MESH_GRAPH_INTEGER, MESH_GRAPH_RESOURCES };
enum { MESH_GRAPH_BITS16, MESH_GRAPH_BITS32, MESH_GRAPH_BITS64, MESH_GRAPH_WIDTHS };
struct mesh_graph_rate { double peak; uint32_t lanes, padding; };
struct mesh_graph_work { double operations[MESH_GRAPH_RESOURCES][MESH_GRAPH_WIDTHS]; uint64_t unknown; };
struct mesh_graph_machine { struct mesh_graph_rate rate[MESH_GRAPH_RESOURCES][MESH_GRAPH_WIDTHS];
                            double bandwidth, block_bandwidth, l2, l1, launch, l2_speedup, l1_speedup, overlap;
                            uint32_t cores, padding; double direct, blit; };
struct mesh_graph_read { uint64_t net, total; double utilization; };
struct mesh_graph_actor { struct mesh_graph_work work; uint64_t blocks, threads, written, launches; size_t reads;
                          const struct mesh_graph_read *read; uint64_t direct, blits; };
int mesh_graph_actor_time(const struct mesh_graph_machine *machine, const struct mesh_graph_actor *actor, double *time);
enum { MESH_GRAPH_COST_ACTOR, MESH_GRAPH_COST_SUM };
struct mesh_graph_cost { uint32_t kind, first, second, padding; };
int mesh_graph_cost_times(const struct mesh_graph_machine *, size_t actors, const struct mesh_graph_actor *,
                          size_t nodes, const struct mesh_graph_cost *, double *times, int *status);
/* design/joint-compilation.md §3, row 5 */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint64_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint64_t *lags);
/* design/joint-compilation.md §3, row 7 */
int mesh_graph_chains(uint32_t items, const uint64_t *count, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, uint64_t *chain, uint64_t *chains);
/* design/joint-compilation.md §3, row 7 */
int mesh_graph_fat_cover(uint64_t length, uint32_t intervals, const uint64_t *start, const uint64_t *end,
                         const uint64_t *need, uint64_t *color, uint64_t *colors);
/* design/joint-compilation.md §3, row 8 */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion);
/* design/unequal-shards-change-request.md items 6 and 14; S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3): the
   shares B_j of `ranks` ranks that minimise, summed over `stages` stages, comm[i] times the largest B_j plus the largest
   over j of fixed[i * ranks + j] + slope[i * ranks + j] * B_j, subject to the B_j summing to 1 and 0 <= B_j <= upper[j] */
int mesh_graph_shares(uint32_t ranks, uint32_t stages, const double *comm, const double *fixed, const double *slope,
                      const double *upper, double *share);
/* design/joint-compilation.md §3, row 8 */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place);
/* M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2, step 3; L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper.
   Res. 22(3), 1997, §2; design/integration-finishing-change-request.md item 2 */
int mesh_graph_heap_cuts(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from, const uint32_t *to,
                         const unsigned char *counted, uint32_t *listed, uint32_t *before, uint32_t *count);
/* design/compilation-composition-change-request.md item 6.3: a placement (place[v] the position of vertex v, a
   permutation of 0 .. vertices - 1, e.g. mesh_graph_heap_order's or mesh_graph_schedule's) restricted to the `kept`
   vertices: at[r] the r-th kept vertex in place order (at has room for `vertices`; its first *count entries are
   these), rank[v] for every v the number of kept vertices placed before it */
void mesh_graph_project(uint32_t vertices, const uint32_t *place, const unsigned char *kept, uint32_t *rank, uint32_t *at,
                        uint32_t *count);
/* design/aot-compilation-change-request.md, appendix item 2, "Its room is fixed at compile time, by the self-timed
   buffer bound"; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997 (BSL) */
/* BSL §3; R. E. Tarjan, SIAM J. Comput. 1(2), 1972 */
int mesh_graph_components(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *component, uint32_t *count);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_cycle_mean(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          const uint64_t *delay, double *mean);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_min_delay(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                         const uint64_t *delay, size_t e, double lambda, uint64_t bound, uint64_t *least);
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_convert_sc(uint32_t vertices, const double *time, size_t edges, const uint32_t *from, const uint32_t *to,
                          uint32_t *added, uint32_t *added_from, uint32_t *added_to);
/* design/torch-capture-change-request.md item 2, "The number of locations of a value the host writes", and its
   2026-11-09 appendix; BSL §4, p. 9, §4.2-4.3, (6) and Lemma 3; M. Lam, PLDI 1988, §7: the node's IPC graph of `actors`
   actors in issue order on two processors, actor j's host part (its CPU work and encoding, t(v) host[j]) vertex j and
   its device part (t(v) work[j]) vertex actors + j, each processor's parts one cycle closed with one delay, the
   feedforward edges e_j from host part j to device part j, and the `edges` dependences `edge` between parts the
   compiled program adds (a CPU part reading a device part's results). bound[j] is Bff(e_j): the least m for which the
   reverse edge d_m(device j, host j) leaves lambda_max (mesh_graph_cycle_mean) unchanged, by mesh_graph_min_delay's
   binary search below (6)'s ceil(sum t / lambda_max); 1 where every t is 0. EINVAL for a time that is negative or not
   finite, an edge outside the graph, or a cycle of no delay */
struct mesh_graph_dependence;
int mesh_graph_buffer_bound(uint32_t actors, const double *host, const double *work, size_t edges,
                            const struct mesh_graph_dependence *edge, uint64_t *bound);
/* design/joint-compilation.md §3, row 6; lambda is mesh_graph_cycle_mean of (time, edges, from, to, delay), BSL Fig. 8's
   λmax of the graph before the added edges (which enter at infinite delay) */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, double lambda, uint64_t *added_delay);
/* design/torch-source-lowering-change-request.md item 3 */
struct mesh_graph_access { uint64_t key, start, end; uint32_t vertex, write; };
/* MLIR affine_map with floordiv and mod: coefficient ((x[dim] floordiv divisor) mod modulus), modulus 0 none, divisor 1
   none (metal_ir.h metal_ir_term, the same record) */
struct mesh_graph_term { uint32_t dim, padding; int64_t coefficient; uint64_t divisor, modulus; };
/* design/compilation-composition-change-request.md, "Complete replacement workgroup", values and effects; J. Lee,
   Ph.D. dissertation, University of Michigan, 2015, Chapter III, §3.2.2: the bytes threadgroup g = (g_0, g_1, g_2) of
   dispatch `dispatch` touches through its binding `binding`, [start + t(g), + extent) of the bound value's storage, t the
   sum of terms [first, first + terms) of the caller's term array over g (metal_ir.h metal_ir_footprints), `exact` 0
   where that is a bound */
struct mesh_graph_footprint { uint32_t dispatch, binding, exact, first, terms, padding; int64_t start; uint64_t extent; };
_Static_assert(sizeof(struct mesh_graph_term) == 32 && sizeof(struct mesh_graph_footprint) == 40,
               "metal_ir.h metal_ir_term, metal_ir_footprint");
/* design/plan-to-bridge-change-request.md R1; torch 2.14.0 c10/core/TensorImpl.h, storage_offset_ and
   sizes_and_strides_; MPI 4.1 §6.1.3, MPI_TYPE_CREATE_SUBARRAY: the bytes [offset + sum_d i_d stride_d, + element) for
   every index i_d < size_d, `pairs` {size, stride} a dimension in bytes, outermost first */
struct mesh_strided { uint64_t offset,element; uint32_t dims,padding; const uint64_t *pairs; };
_Static_assert(sizeof(struct mesh_strided)==32 && offsetof(struct mesh_strided,dims)==16 &&
               offsetof(struct mesh_strided,pairs)==24,"mesh_strided: rdma/mesh.py Strided");
/* design/compilation-composition-change-request.md, "Complete replacement workgroup" (exact effects without
   enumeration); S. Verdoolaege, isl: An Integer Set Library for the Polyhedral Model, ICMS 2010, LNCS 6327 (isl 0.27):
   an access of vertex `vertex` to the view's bytes of storage `key`, `write` as struct mesh_graph_access's. Two views
   of one key, of different vertices, at least one written (both 2 do not conflict), conflict exactly where some byte
   is in both: the integer points of offset_a + sum_k i_k stride_k + u = offset_b + sum_l j_l stride_l + v over their
   bounded indices and element bytes u, v, isl_basic_set_is_empty (exact, arbitrary precision), no index or byte
   enumerated and no hull taken for a conflict. The relation keeps each edge's
   causes (design/compilation-composition-change-request.md, "Retain the value/effect/transfer cause of an edge", item
   6.4): one record {from, to} a distinct (from, to, cause, witness), sorted by from, to, kind, key, given, from_view,
   to_view, `from` < `to`; a conflict's cause its key and hazard (the `from` access before the `to` access:
   READ_AFTER_WRITE, WRITE_AFTER_READ, WRITE_AFTER_WRITE), its witness the two meeting views, from_view `from`'s and
   to_view `to`'s, indices into `view` as given; supplied edge g's (supplied_from[g] < supplied_to[g], EINVAL
   otherwise) GIVEN with `given` g and both witnesses SIZE_MAX. *conflict holds *count records, the caller's, released
   with free (NULL and 0 on failure). A pair of views of one vertex (mesh_graph_conflicts_views) is one record per
   hazard, READ_AFTER_WRITE's from_view the written view and WRITE_AFTER_READ's the read one. The pure relation is its
   distinct (from, to) */
struct mesh_graph_view { uint64_t key; struct mesh_strided view; uint32_t vertex, write; };
_Static_assert(sizeof(struct mesh_graph_view) == 48, "mesh_graph_view");
enum { MESH_GRAPH_READ_AFTER_WRITE, MESH_GRAPH_WRITE_AFTER_READ, MESH_GRAPH_WRITE_AFTER_WRITE, MESH_GRAPH_GIVEN };
struct mesh_graph_cause { uint64_t key; uint32_t kind, given; size_t from_view, to_view; };
_Static_assert(sizeof(struct mesh_graph_cause) == 32 && offsetof(struct mesh_graph_cause, from_view) == 16 &&
               offsetof(struct mesh_graph_cause, to_view) == 24, "mesh_graph_cause: metal_ir.h metal_ir_dependences");
struct mesh_graph_conflict { uint32_t from, to, kind, given; uint64_t key; size_t from_view, to_view; };
_Static_assert(sizeof(struct mesh_graph_conflict) == 40 && offsetof(struct mesh_graph_conflict, from_view) == 24 &&
               offsetof(struct mesh_graph_conflict, to_view) == 32, "mesh_graph_conflict: rdma/mesh_graph.py Conflict");
/* design/compilation-composition-change-request.md: retained exact intersections, including self conflicts;
   caller owns *conflict and releases it with free. */
int mesh_graph_conflicts_views(size_t views, const struct mesh_graph_view *view, size_t supplied,
                               const uint32_t *supplied_from, const uint32_t *supplied_to,
                               struct mesh_graph_conflict **conflict, size_t *count);
int mesh_graph_dependences_views(size_t views, const struct mesh_graph_view *view, size_t supplied,
                                 const uint32_t *supplied_from, const uint32_t *supplied_to,
                                 struct mesh_graph_conflict **conflict, size_t *count);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §7, Fig. 4 */
int mesh_graph_redundant(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, const uint64_t *delay,
                         const unsigned char *synchronization, unsigned char *redundant);
/* design/torch-source-lowering-change-request.md item 3; design/compilation-composition-change-request.md item 6.1:
   the dependence relation (from, to) is mesh_graph_dependences_views' distinct (from, to), edge e its identity.
   redundant[e] marks an edge another path covers (mesh_graph_redundant at zero delays, every edge a candidate); kept
   counts the others */
int mesh_graph_synchronization(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                               unsigned char *redundant, size_t *kept);
/* design/torch-source-lowering-change-request.md item 3; design/compilation-composition-change-request.md item 6.3:
   levels and places over the caller's dependence relation (from[e] < to[e]); level the longest path to v, place the
   heap order by -level (mesh_graph_heap_order) */
int mesh_graph_schedule(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, uint32_t *level,
                        uint32_t *place);
int mesh_graph_levels(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, uint32_t *level);
/* design/synchronization-signature-change-request.md items 1, 2, 4 and 6; S. S. Bhattacharyya, S. Sriram and E. A. Lee,
   IEEE Trans. Signal Processing 45(6), 1997, §4.1, §5.1 and §7, Fig. 4; E. A. Lee and D. G. Messerschmitt, IEEE Trans.
   Computers C-36(1), 1987; M. Lam, PLDI 1988; L. Lamport, CACM 21(7), 1978 */
#define MESH_GRAPH_NONE UINT32_MAX
enum { MESH_GRAPH_QUEUE, MESH_GRAPH_HOST, MESH_GRAPH_BRIDGE, MESH_GRAPH_KERNEL, MESH_GRAPH_KINDS };
enum { MESH_GRAPH_SUBMIT, MESH_GRAPH_QUEUE_SIGNAL, MESH_GRAPH_QUEUE_WAIT, MESH_GRAPH_FENCE_UPDATE, MESH_GRAPH_FENCE_WAIT,
       MESH_GRAPH_PASS_BARRIER, MESH_GRAPH_QUEUE_BARRIER, MESH_GRAPH_HOST_WAIT, MESH_GRAPH_HOST_NOTIFY,
       MESH_GRAPH_BRIDGE_START, MESH_GRAPH_THREADGROUP_BARRIER, MESH_GRAPH_SIMDGROUP_BARRIER, MESH_GRAPH_ATOMIC,
       MESH_GRAPH_PRODUCER_BARRIER, MESH_GRAPH_BRIDGE_RAISE, MESH_GRAPH_INITIAL, MESH_GRAPH_HOST_PUBLISH,
       MESH_GRAPH_KERNEL_FENCE, MESH_GRAPH_HOST_SIGNAL, MESH_GRAPH_PRIMITIVES };
enum { MESH_GRAPH_IMAGE, MESH_GRAPH_STRONGER, MESH_GRAPH_FALSE, MESH_GRAPH_UNMATCHED, MESH_GRAPH_SATISFIED,
       MESH_GRAPH_TRANSFORMED, MESH_GRAPH_STATES };
struct mesh_graph_processor { uint32_t kind, index; };
/* one program's identities across its producers: a processor's `index` is thread * MESH_GRAPH_NODES + node (the host's
   program thread 0 and its event listener 1; a queue's ordinal; the bridge 0), node its map index; a plan count word
   is object MESH_GRAPH_PLAN_COUNT + its added edge, a compiled function's submissions MESH_GRAPH_PLAN_SUBMISSION + its
   index in mesh_plan's `functions`, each submission's value the plan's executions after it (mesh_transfers_start) */
#define MESH_GRAPH_NODES 64u
#define MESH_GRAPH_PLAN_COUNT UINT32_C(0x80000000)
#define MESH_GRAPH_PLAN_SUBMISSION UINT32_C(0xC0000000)
#define MESH_GRAPH_LOWERED UINT32_C(0x40000000)
struct mesh_graph_task { uint32_t processor, place, label, copy; };
struct mesh_graph_dependence { uint32_t from, to; uint64_t delay; };
struct mesh_graph_operation { uint32_t processor, primitive, vertex, object; uint64_t execution, value; };
struct mesh_graph_synchronized { uint32_t signal, wait, state, redundant, dependence, witness; uint64_t delay; };
_Static_assert(sizeof(struct mesh_graph_processor) == 8 && sizeof(struct mesh_graph_task) == 16 &&
               sizeof(struct mesh_graph_dependence) == 16 && sizeof(struct mesh_graph_operation) == 32 &&
               sizeof(struct mesh_graph_synchronized) == 32, "rdma/mesh_graph.py");
/* design/algorithm-sources.md#relations-over-copies-and-classes; design/compilation-composition-change-request.md item 6.2;
   E. W. Dijkstra, Numerische Mathematik 1, 1959. mesh_graph_contract, mesh_graph_contract_unrolled and
   mesh_graph_quotient return their relation once: *made (*contracted) holds *count (*made) records, the caller's,
   released with free (NULL and 0 on failure) */
int mesh_graph_contract(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                        const unsigned char *kept, struct mesh_graph_dependence **contracted, size_t *made);
/* design/algorithm-sources.md#relations-over-copies-and-classes; M. Lam, PLDI 1988, §2.3 */
int mesh_graph_contract_unrolled(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                                 const unsigned char *kept, uint32_t copies, struct mesh_graph_dependence **contracted,
                                 size_t *made);
/* design/algorithm-sources.md#relations-over-copies-and-classes;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §4.1 */
int mesh_graph_quotient(uint32_t vertices, const uint32_t *class_of, size_t edges, const uint32_t *from, const uint32_t *to,
                        struct mesh_graph_dependence **made, size_t *count);
/* design/compilation-composition-change-request.md items 6, 6.4 and 7, "Retain the value/effect/transfer cause of an
   edge": one retained relation, reference counted. A root borrows the caller's views (and their pairs) and
   conflicts unchanged under `owner`, adopted on success and released by release(owner) with the last reference
   (NULL release: the caller keeps them alive). Its edges are the distinct (from, to), from != to, of the conflicts,
   sorted, delay 0; edge e's causes are conflict[member[first[e] .. first[e + 1])], the original records (a self
   conflict is retained and belongs to no edge). row_first/row_edge index the edges by `from` (outgoing) and
   in_first/in_edge by `to` (incoming): row_edge[row_first[v] .. row_first[v + 1]) are v's. A derived relation
   retains its parent and adopts its operand's owner: mesh_graph_relation_quotient is mesh_graph_quotient's relation
   over classes, edge e from every parent edge source[source_first[e] .. source_first[e + 1])] between its two
   classes (edges inside one class stay in the parent); mesh_graph_relation_contract is mesh_graph_contract's,
   edge e's sources the parent edges of the least-delay walk mesh_graph_lags selected, in walk order. A derived
   relation's view and conflict arrays are the root's, and its members compose its sources' members; class_of (a
   quotient's) and kept (a contraction's) are its operand as given, over the parent's vertices, NULL otherwise. Data is
   borrowed until the relation is freed. mesh_graph_relation_synchronization is mesh_graph_synchronization's mask
   over the relation's edges, computed once. EINVAL for an endpoint or class out of range */
struct mesh_graph_relation;
struct mesh_graph_relation_data {
  uint32_t vertices;
  size_t views, conflicts, edges;
  const struct mesh_graph_view *view;
  const struct mesh_graph_conflict *conflict;
  const struct mesh_graph_dependence *edge;
  const size_t *first, *member;
  const size_t *row_first, *row_edge;
  const size_t *in_first, *in_edge;
  const struct mesh_graph_relation *parent;
  const size_t *source_first, *source;
  const uint32_t *class_of;
  const unsigned char *kept;
};
void mesh_graph_relation_retain(struct mesh_graph_relation *);
void mesh_graph_relation_free(struct mesh_graph_relation *);
int mesh_graph_relation_create(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t conflicts,
                               const struct mesh_graph_conflict *conflict, void (*release)(void *), void *owner,
                               struct mesh_graph_relation **out);
const struct mesh_graph_relation_data *mesh_graph_relation_data(const struct mesh_graph_relation *);
int mesh_graph_relation_quotient(struct mesh_graph_relation *parent, uint32_t classes, const uint32_t *class_of,
                                 void (*release)(void *), void *owner, struct mesh_graph_relation **out);
int mesh_graph_relation_contract(struct mesh_graph_relation *parent, const unsigned char *kept, void (*release)(void *),
                                 void *owner, struct mesh_graph_relation **out);
int mesh_graph_relation_synchronization(struct mesh_graph_relation *, const unsigned char **redundant, size_t *kept);
enum { MESH_GRAPH_COMPLETION_EVENT = 1 };
struct mesh_graph_event { uint32_t object, role, argument, padding; uint64_t lookback; };
struct mesh_graph_span { uint64_t first, count, begin, end; };
/* design/synchronization-signature-change-request.md item 1; E. A. Lee and D. G. Messerschmitt, Proc. IEEE 75(9), 1987
   (delays); M. Lam, PLDI 1988, section 2.3 (unrolled copies): a producer's operation repeated once per execution K, its
   static fields `processor`, `primitive`, `vertex`, `object`, its primitive INITIAL, HOST_WAIT, BRIDGE_START or
   BRIDGE_RAISE (rdma/mesh-collective.c plan_signature's). INITIAL once, before every repeated row; HOST_WAIT value
   max(K every + ordinal - lookback every, 0); BRIDGE_START value K every + ordinal; BRIDGE_RAISE submitted = K every +
   ordinal - 1, value submitted + 1, at `vertex` and execution (submitted - lookback) / every where submitted >=
   lookback, else at `startup` and K; each repeated row at execution K otherwise, rows in their order within each K */
struct mesh_graph_repeated { uint32_t processor, primitive, vertex, object; uint64_t every, ordinal, lookback;
                             uint32_t startup, padding; };
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §9: added edge `witness`
   from task `from` to `to` at pair delay `total_delay` (its delay plus its tail's offset), `proven` its
   mesh_graph_resynchronized test; sorted by from, to, witness */
struct mesh_graph_proof { uint32_t from, to; uint64_t total_delay; uint32_t witness, proven; };
_Static_assert(sizeof(struct mesh_graph_repeated) == 48 && sizeof(struct mesh_graph_proof) == 24,
               "rdma/mesh_graph.py Repeated, Proof");
struct mesh_graph_signature { uint32_t processors, tasks; size_t operations, dependences, synchronized, uncovered,
                              redundant, unwaited, states[MESH_GRAPH_STATES];
                              struct mesh_graph_processor *processor; struct mesh_graph_task *task;
                              struct mesh_graph_operation *operation; struct mesh_graph_dependence *dependence;
                              struct mesh_graph_synchronized *edge; uint64_t *position, *count, *least;
                              size_t *previous, *waited, *first, *order; uint64_t *unknown; uint32_t *alias;
                              unsigned char digest[32]; size_t events; struct mesh_graph_event *event;
                              size_t spans; struct mesh_graph_span *span; size_t repeats;
                              struct mesh_graph_repeated *repeated; size_t proofs; struct mesh_graph_proof *proof; };
int mesh_graph_signature(uint32_t processors, const struct mesh_graph_processor *processor, uint32_t tasks,
                         const struct mesh_graph_task *task, size_t dependences,
                         const struct mesh_graph_dependence *dependence, size_t operations,
                         const struct mesh_graph_operation *operation, struct mesh_graph_signature *signature);
void mesh_graph_signature_free(struct mesh_graph_signature *signature);
/* design/synchronization-signature-change-request.md item 1; design/compilation-composition-change-request.md items 3
   and 6.4: mesh_graph_signature over the rows' operations for executions [0, executions), the rows kept on the
   signature. mesh_graph_signature_span realizes a kept signature's rows for `executions` the same way and applies its
   kept proofs as mesh_graph_resynchronized does; ENODATA for a kept signature with operations and no rows (a record
   written before rows were kept: its repeated operands are absent) */
int mesh_graph_signature_repeated(uint32_t processors, const struct mesh_graph_processor *processor, uint32_t tasks,
                                  const struct mesh_graph_task *task, size_t dependences,
                                  const struct mesh_graph_dependence *dependence, size_t repeats,
                                  const struct mesh_graph_repeated *repeated, uint64_t executions,
                                  struct mesh_graph_signature *signature);
int mesh_graph_signature_span(const struct mesh_graph_signature *kept, uint64_t executions,
                              struct mesh_graph_signature *signature);
int mesh_graph_signature_program(struct mesh_graph_signature *, size_t events, const struct mesh_graph_event *);
int mesh_graph_signature_spans(struct mesh_graph_signature *, size_t spans, const struct mesh_graph_span *);

/* design/synchronization-signature-change-request.md item 1, "its SHA-256 is part of the compiled artifact": the
   signature's one serialized form, its header then processor, task, dependence, operation, edge, least and alias in
   that order; mesh_graph_record_read derives the rest as mesh_graph_signature does */
struct mesh_graph_record { unsigned char digest[32]; uint64_t bytes; uint32_t processors, tasks; uint64_t operations,
                           dependences, synchronized; };
_Static_assert(sizeof(struct mesh_graph_record) == 72, "mesh_graph_record");
int mesh_graph_record(const struct mesh_graph_signature *signature, void *record, size_t room, size_t *made);
int mesh_graph_record_read(const void *record, struct mesh_graph_signature *signature);
int mesh_graph_archive(const struct mesh_graph_signature *, void *bytes, size_t room, size_t *made);
int mesh_graph_archive_read(const void *bytes, size_t length, struct mesh_graph_signature *);
/* design/synchronization-signature-change-request.md item 1; L. Lamport, CACM 21(7), 1978: one program's signature from
   two producers' (a: the plan's, b: the replay's). Processors are united by identity, a's tasks and dependences come
   first and b's after, each producer's pairs keep their states, witnesses and dependences, and a native operation
   both list (BRIDGE_START, same processor identity, object and value) is b's, its `alias` a's task, with a's operations
   on that processor before it in a's order placed before it */
int mesh_graph_compose(const struct mesh_graph_signature *a, const struct mesh_graph_signature *b,
                       struct mesh_graph_signature *signature);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §9, Theorem 4, Fig. 6
   (Convert-to-SC-graph) and Fig. 8 (DetermineDelays, MinDelay); design/aot-compilation-change-request.md, appendix
   item 2, BS-4. The program graph (time, edges, from, to, delay) holds no emitted synchronization; `added` edges
   (added_from, added_to, added_delay) and `lambda` are the one Convert-to-SC and DetermineDelays pass's output. Edge i
   is witnessed where its ends lie in different strongly connected components of the program graph, the added edges
   make it strongly connected and edge i is necessary to that, and its delay is MinDelay's: with the earlier edges at
   their delays and the later ones at none, no cycle's mean exceeds lambda at added_delay[i] and one does at one less.
   A `false` pair whose task edge is edge i and whose delay less `offset[tail]` is added_delay[i] has `witness` i, and
   is `transformed` where edge i is witnessed; each edge's result is kept as one of the signature's `proof` */
int mesh_graph_resynchronized(struct mesh_graph_signature *signature, const double *time, size_t edges,
                              const uint32_t *from, const uint32_t *to, const uint64_t *delay, uint32_t added,
                              const uint32_t *added_from, const uint32_t *added_to, const uint64_t *added_delay,
                              double lambda, const uint64_t *offset);
/* R. F. Boisvert, R. Pozo and K. A. Remington, The Matrix Market Exchange Formats: Initial Design, NIST IR 5935, 1996 */
int mesh_graph_matrix_market(const struct mesh_graph_signature *signature, char *text, size_t room, size_t *made);
/* Graphviz, The DOT Language, subgraph cluster; E. R. Gansner, E. Koutsofios, S. C. North and K.-P. Vo, IEEE TSE 19(3),
   1993; A. V. Aho, M. R. Garey and J. D. Ullman, SIAM J. Comput. 1(2), 1972 */
int mesh_graph_dot(const struct mesh_graph_signature *signature, char *text, size_t room, size_t *made);
/* L. Lamport, CACM 21(7), 1978, Fig. 1; git graph.c, "Draw a text-based graphical representation of the commit
   history"; A. B. Kahn, CACM 5(11), 1962 */
int mesh_graph_space_time(const struct mesh_graph_signature *signature, char *text, size_t room, size_t *made);
/* MPI 4.1 §6.1.11 */
struct mesh_graph_interval { uint64_t key, start, end; };
int mesh_graph_intersect(size_t count, struct mesh_graph_interval *interval);
/* GaijinEntertainment/DagorEngine 2baf1947, prog/gameLibs/render/daFrameGraph/backend/resourceScheduling/heapLayout.h:19
   and heapLayout.cpp:13-23, do_arcs_overlap */
int mesh_graph_arcs_overlap(uint64_t a_start, uint64_t a_end, uint64_t b_start, uint64_t b_end);
/* TensorFlow Lite 6cf18d88, tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.h:43-45,
   GreedyBySizeAssignment; design/torch-capture-change-request.md item 6 */
int mesh_graph_greedy_by_size(size_t records, const uint64_t *size, int (*intersect)(const void *context, size_t a, size_t b),
                              const void *context, uint64_t align, uint64_t *offset, uint64_t *total);

void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §5.2; B. R. Rau, MICRO 1994;
   design/torch-source-lowering-change-request.md item 3 */
size_t mesh_graph_delayed(const size_t *first, const size_t *order, const uint32_t *tail, uint32_t row, uint32_t count,
                          uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot, size_t *edge);
/* design/compilation-composition-change-request.md item 6: the producer execution e a consumer's execution
   e_b = b every_b + ordinal_b reads, e = (e_b / every) producer + ordinal in 64 bits, every = every_b, producer the
   producing function's every_f, ordinal the producing call's ordinal_f; {1, 1, 0} the consumer's own (e = e_b). Its
   readiness: mesh-metal.h struct mesh_metal_readiness, compiled by metal-microbench metal_transport.mm */
struct mesh_graph_generation { uint32_t every, producer, ordinal; };

#ifdef __cplusplus
}
#endif
#endif
