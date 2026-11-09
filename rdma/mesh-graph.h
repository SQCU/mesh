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
/* design/torch-capture-change-request.md, "Source for item 2's execution-time estimates"; OpenXLA,
   xla/service/gpu/model/gpu_performance_model_base.{h,cc} and gpu_performance_model.cc (read 2026-11-09):
   CalculateEffectiveFlopsPerNs, ComputeTime, ReadTimeWithDRAMHeuristic, AdjustBandwidth, WriteTime,
   CombineComputeAndMemoryAccessTime, EstimateRunTimeForInstructionImpl (read time summed over the operands),
   kL2CacheSpeedup 2.5, kL1CacheSpeedup 8, kMemoryComputeParallelism 0.95, kKernelLaunchOverhead per launch; S. Williams,
   A. Waterman and D. Patterson, Roofline, CACM 52(4), 2009, §3-4: one machine's operands (a resource's clock in GHz,
   cores, FPUs a core, DRAM bandwidth in bytes/s, L2 bytes, L1 bytes a core, memory transactions a clock, launch
   overhead in ns), from its specifications or measurements. An actor's estimate t(v) in ns: its flops over `blocks`
   blocks of `threads` threads, each operand read `net` distinct and `total` accessed bytes at its coalescing
   `utilization`, `written` bytes, `launches` kernel launches */
struct mesh_graph_machine { double clock, bandwidth, l2, l1, transactions, launch; uint32_t cores, fpus; };
struct mesh_graph_read { uint64_t net, total; double utilization; };
struct mesh_graph_actor { double flops; uint64_t blocks, threads, written, launches; size_t reads;
                          const struct mesh_graph_read *read; };
double mesh_graph_actor_time(const struct mesh_graph_machine *machine, const struct mesh_graph_actor *actor);
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
/* design/compilation-composition-change-request.md item 6.2: the relation is the supplied edges (supplied_from[e] ->
   supplied_to[e]) united with the accesses' conflicts (lower vertex first), each pair once, sorted by (from, to); every
   edge runs forward in the caller's issue numbering (EINVAL for a supplied edge with from >= to: it would wait on work
   issued after it); ERANGE with *edges the count where it exceeds room. `access` is sorted in place */
int mesh_graph_dependences(size_t accesses, struct mesh_graph_access *access, size_t supplied,
                           const uint32_t *supplied_from, const uint32_t *supplied_to, size_t room, uint32_t *from,
                           uint32_t *to, size_t *edges);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §7, Fig. 4 */
int mesh_graph_redundant(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, const uint64_t *delay,
                         const unsigned char *synchronization, unsigned char *redundant);
/* design/torch-source-lowering-change-request.md item 3; design/compilation-composition-change-request.md item 6.1:
   the dependence relation (from, to) is mesh_graph_dependences' result, edge e its identity. redundant[e] marks an
   edge another path covers (mesh_graph_redundant at zero delays, every edge a candidate); kept counts the others */
int mesh_graph_synchronization(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                               unsigned char *redundant, size_t *kept);
/* design/torch-source-lowering-change-request.md item 3; design/compilation-composition-change-request.md item 6.3:
   levels and places over the caller's dependence relation (from[e] < to[e]); level is the longest path, with program
   order (unless self_timed) from the previous vertex in program order to v weighted after[v] (0 or 1); a vertex with
   after[v] MESH_GRAPH_UNORDERED is a lowered vertex outside program order, placed by its edges alone; place the heap
   order by -level if self_timed, else identity */
#define MESH_GRAPH_UNORDERED 2
int mesh_graph_schedule(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                        const unsigned char *after, int self_timed, uint32_t *level, uint32_t *place);
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
struct mesh_graph_task { uint32_t processor, place, label, copy; };
struct mesh_graph_dependence { uint32_t from, to; uint64_t delay; };
struct mesh_graph_operation { uint32_t processor, primitive, vertex, object; uint64_t execution, value; };
struct mesh_graph_synchronized { uint32_t signal, wait, state, redundant, dependence, witness; uint64_t delay; };
_Static_assert(sizeof(struct mesh_graph_processor) == 8 && sizeof(struct mesh_graph_task) == 16 &&
               sizeof(struct mesh_graph_dependence) == 16 && sizeof(struct mesh_graph_operation) == 32 &&
               sizeof(struct mesh_graph_synchronized) == 32, "rdma/mesh_graph.py");
/* design/compilation-composition-change-request.md item 6.2; E. W. Dijkstra, 1959 (mesh_graph_lags): the relation
   (edge, delays in executions) with every vertex v not `kept[v]` contracted. For kept u and w, one output {u, w, d} where
   a walk u -> ... -> w of at least one edge has every interior vertex contracted, d the least sum of its delays (so a
   kept -> kept edge is its own walk, and chains of any length through contracted vertices accumulate); u = w is a
   cycle through contracted vertices. Ordered by (u, w); ERANGE with *made the count where it exceeds room (at most
   kept vertices squared). The least delay alone implies the larger ones only where u's executions are ordered among
   themselves; where they may overlap, the caller unrolls the executions as vertices with delay 0 (metal_recording.m
   MetalPlannedOrder), so every copy pair is its own output */
int mesh_graph_contract(uint32_t vertices, size_t edges, const struct mesh_graph_dependence *edge,
                        const unsigned char *kept, size_t room, struct mesh_graph_dependence *contracted, size_t *made);
struct mesh_graph_signature { uint32_t processors, tasks; size_t operations, dependences, synchronized, uncovered,
                              redundant, unwaited, states[MESH_GRAPH_STATES];
                              struct mesh_graph_processor *processor; struct mesh_graph_task *task;
                              struct mesh_graph_operation *operation; struct mesh_graph_dependence *dependence;
                              struct mesh_graph_synchronized *edge; uint64_t *position, *count, *least;
                              size_t *previous, *waited, *first, *order; uint64_t *unknown; uint32_t *alias;
                              unsigned char digest[32]; };
int mesh_graph_signature(uint32_t processors, const struct mesh_graph_processor *processor, uint32_t tasks,
                         const struct mesh_graph_task *task, size_t dependences,
                         const struct mesh_graph_dependence *dependence, size_t operations,
                         const struct mesh_graph_operation *operation, struct mesh_graph_signature *signature);
void mesh_graph_signature_free(struct mesh_graph_signature *signature);
/* design/synchronization-signature-change-request.md item 1, "its SHA-256 is part of the compiled artifact": the
   signature's one serialized form, its header then processor, task, dependence, operation, edge, least and alias in
   that order; mesh_graph_record_read derives the rest as mesh_graph_signature does */
struct mesh_graph_record { unsigned char digest[32]; uint64_t bytes; uint32_t processors, tasks; uint64_t operations,
                           dependences, synchronized; };
_Static_assert(sizeof(struct mesh_graph_record) == 72, "mesh_graph_record");
int mesh_graph_record(const struct mesh_graph_signature *signature, void *record, size_t room, size_t *made);
int mesh_graph_record_read(const void *record, struct mesh_graph_signature *signature);
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
   is `transformed` where edge i is witnessed */
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
                          uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot);
/* design/compilation-composition-change-request.md item 6: the producer execution e a consumer's execution
   e_b = b every_b + ordinal_b reads, e = (e_b / every) producer + ordinal in 64 bits, every = every_b, producer the
   producing function's every_f, ordinal the producing call's ordinal_f; {1, 1, 0} the consumer's own (e = e_b). Its
   readiness: mesh-metal.h struct mesh_metal_readiness, compiled by metal-microbench metal_transport.mm */
struct mesh_graph_generation { uint32_t every, producer, ordinal; };

#ifdef __cplusplus
}
#endif
#endif
