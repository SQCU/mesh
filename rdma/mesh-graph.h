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
   §22.3 and §22.4, Lemma 22.11 */
int mesh_graph_tree_visit(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                          const void *context, uint32_t root,
                          int (*visit)(void *consumer, uint32_t vertex, uint32_t parent), void *consumer);
/* T. H. Cormen, C. E. Leiserson, R. L. Rivest and C. Stein, Introduction to Algorithms, 3rd ed., MIT Press, 2009, §22.1 */
void mesh_graph_neighbours(size_t edges, uint32_t (*end)(const void *context, size_t e, uint32_t side), const void *context,
                           uint32_t from, void (*visit)(void *consumer, uint32_t to), void *consumer);
int mesh_graph_depth(uint32_t vertices, const size_t *first, const uint32_t *head, const unsigned char *marked, uint32_t *depth);
/* Y. J. Chu and T. H. Liu, Scientia Sinica 14, 1965; J. Edmonds, J. Res. Nat. Bur. Standards 71B, 1967 */
int mesh_graph_arborescence(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                            const double *weight, int32_t *chosen);
int mesh_graph_arborescence_within(uint32_t N, uint32_t root, uint32_t E, const uint32_t *from, const uint32_t *to,
                                   const double *weight, uint32_t depth, int32_t *chosen);
uint32_t mesh_graph_climb(const uint32_t *parent, uint32_t vertices, uint32_t root, uint32_t from,
                          void (*visit)(void *consumer, uint32_t at, uint32_t step), void *consumer);
uint32_t mesh_graph_cycle_length(const uint32_t *parent, uint32_t vertices, uint32_t from);
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
void mesh_graph_tree_loads(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                           const double *weight, double *load);
uint32_t mesh_graph_grid_cut(uint64_t first, uint64_t length, uint64_t cell, uint64_t *start, uint64_t *count, uint64_t *index);
uint32_t mesh_graph_tree_depth(uint32_t vertices, uint32_t root, const uint32_t *parent);
double mesh_graph_trees_rate(uint32_t vertices, uint32_t trees, const uint32_t *root, const uint32_t *parent,
                             const double *weight, const double *capacity);
/* design/joint-compilation.md §3, row 4 */
double mesh_graph_fill(uint32_t parts, const double *latency, const double *rate, uint64_t units, uint64_t *count);
/* design/joint-compilation.md §3, row 10 */
double mesh_graph_list_schedule(uint32_t machines, const double *rate, uint64_t units, const uint64_t *bytes,
                                uint32_t *machine, uint64_t *carried);
/* design/joint-compilation.md §3, row 5 */
int mesh_graph_lags(uint32_t vertices,const size_t *first,uint32_t (*head)(const void *context,size_t k),
                    uint32_t (*lag)(const void *context,size_t k),const void *context,uint32_t sources,
                    const uint32_t *source,uint32_t targets,const uint32_t *target,uint32_t *lags);
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
/* design/joint-compilation.md §3, row 6 */
int mesh_graph_determine_delays(uint32_t vertices, const double *time, size_t edges, const uint32_t *from,
                                const uint32_t *to, const uint64_t *delay, uint32_t added, const uint32_t *added_from,
                                const uint32_t *added_to, uint64_t *added_delay);
/* design/torch-source-lowering-change-request.md item 3 */
struct mesh_graph_access { uint64_t key, start, end; uint32_t vertex, write; };
int mesh_graph_dependences(size_t accesses, struct mesh_graph_access *access, size_t room, uint32_t *from, uint32_t *to,
                           size_t *edges);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §7, Fig. 4 */
int mesh_graph_redundant(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to, const uint64_t *delay,
                         const unsigned char *synchronization, unsigned char *redundant);
/* design/torch-source-lowering-change-request.md item 3 */
int mesh_graph_synchronization(uint32_t vertices, size_t accesses, struct mesh_graph_access *access, size_t room,
                               uint32_t *from, uint32_t *to, size_t *edges, size_t *dependences);
/* design/torch-source-lowering-change-request.md item 3 */
int mesh_graph_schedule(uint32_t vertices, size_t accesses, struct mesh_graph_access *access, const unsigned char *after,
                        int self_timed, uint32_t *level, uint32_t *place, size_t *dependences, size_t *synchronizations);
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
/* G. E. Blelloch, Prefix sums and their applications, 1990; M. Harris, S. Sengupta and J. D. Owens, GPU Gems 3, ch. 39,
   2007; R. Nasre, M. Burtscher and K. Pingali, IPDPS 2013 */
int mesh_graph_compaction_source(uint32_t items, uint32_t simds, char *text, size_t room);
/* R. Nasre, M. Burtscher and K. Pingali, IPDPS 2013; Arvind, R. S. Nikhil and K. K. Pingali, ACM TOPLAS 11(4), 1989;
   RISC-V V 1.0, the mask register v0; R. M. Russell, CACM 21(1), 1978, the vector mask register; Metal Shading Language
   Specification 4.1, §6.10.1 and §6.10.2 simd_ballot; MTL4ComputeCommandEncoder.h:90-112;
   design/torch-capture-change-request.md item 3 */
int mesh_graph_worklist_source(uint32_t later, const uint32_t groups[3], const uint32_t block[3], const uint32_t *extent,
                               uint32_t simds, uint32_t operands, uint32_t classes, const uint32_t *slots,
                               const char *const builtin[5], char *parameters, size_t parameters_room, char *code,
                               size_t code_room);

#ifdef __cplusplus
}
#endif
#endif
