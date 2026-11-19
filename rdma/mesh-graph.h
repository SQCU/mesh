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
/* UINT32_MAX heads are outside the supplied row view. */
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
   vertex), place[v] v's index in preorder (UINT32_MAX unvisited), end[v] the exclusive end in preorder of v's
   depth-first subtree, preorder[place[v] .. end[v]) (UINT32_MAX unvisited; NULL breadth-first); freed by
   mesh_graph_search_free */
struct mesh_graph_search {
  uint32_t vertices, count;
  uint32_t *preorder, *postorder, *parent, *place;
  size_t *entry;
  uint32_t *end;
};
/* T. H. Cormen et al., 3rd ed., §22.3 and §22.4, Lemma 22.11: depth-first search over rows (first, head) from
   root[0 .. roots) in that order, each row's entries in order (in reverse order where `reverse`), each reachable vertex
   visited once; EDEADLK on a back edge (a cycle), EINVAL on a vertex out of range */
int mesh_graph_search(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                      const void *context, uint32_t roots, const uint32_t *root, int reverse,
                      struct mesh_graph_search **made);
/* T. H. Cormen et al., 3rd ed., §22.2: breadth-first search over rows (first, head) from start[0 .. starts), each start
   and then each row entry in order, a vertex entering once; EINVAL on a vertex out of range. mesh_graph_breadth_masked
   is the same search where active[v] (NULL: every vertex) is nonzero, the row of a vertex v with active[v] == 0 not
   followed (v itself still visited) */
int mesh_graph_breadth(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                       const void *context, uint32_t starts, const uint32_t *start, struct mesh_graph_search **made);
int mesh_graph_breadth_masked(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                              const void *context, uint32_t starts, const uint32_t *start, const unsigned char *active,
                              struct mesh_graph_search **made);
/* nonzero where vertex `vertex` is in the depth-first subtree of `ancestor` (place[ancestor] <= place[vertex] <
   end[ancestor], a vertex in its own subtree); 0 for a breadth-first search (end NULL) or a vertex out of range or
   unvisited */
int mesh_graph_ancestor(const struct mesh_graph_search *search, uint32_t ancestor, uint32_t vertex);
void mesh_graph_search_free(struct mesh_graph_search *made);
/* a search as produced, without searching again: mesh_graph_search_save writes its arrays into *bytes (*length of
   them, malloc'd, the caller's to free): vertices, count and whether it is depth-first (postorder and end), then
   preorder and postorder [0 .. count), parent, place and end [0 .. vertices) (u32 each), entry [0 .. vertices) (u64,
   SIZE_MAX as UINT64_MAX); mesh_graph_search_load makes a search of its own (mesh_graph_search_free) from exactly those
   bytes, EINVAL where they are not one saved search (their length, a count, or an index past what it indexes), ENOMEM */
int mesh_graph_search_save(const struct mesh_graph_search *search, void **bytes, size_t *length);
int mesh_graph_search_load(const void *bytes, size_t length, struct mesh_graph_search **made);
/* rows held as an array: ((const uint32_t *)context)[k] */
uint32_t mesh_graph_row_head(const void *context, size_t k);
/* rows of a retained relation (context its struct mesh_graph_relation_data, first its row_first): edge[row_edge[k]].to */
uint32_t mesh_graph_relation_head(const void *context, size_t k);
/* incoming rows of a retained relation (context its struct mesh_graph_relation_data, first its in_first):
   edge[in_edge[k]].from */
uint32_t mesh_graph_relation_incoming_head(const void *context, size_t k);
/* A. Goralčíková and V. Koubek, LNCS 74, 1979; K. Simon, TCS 58, 1988: over rows (first, head), the vertices processed
   in the order order[0], order[step], ..., order[(vertices - 1) * step] (each vertex once; a search's postorder over the
   same rows, or its reverse with step -1 from the last element), every head before its row in that order: bit u of
   reached[v * words + u / 64] set where a path of rows leads from v to u, words = (vertices + 63) / 64; EINVAL where the
   order is not a permutation of the vertices or a head is not before its row; freed by mesh_graph_closure_free.
   mesh_graph_closure_labels is the same pass over the same rows and order with `columns` label columns, words =
   (columns + 63) / 64: for v with active[v] nonzero (NULL: every vertex) reached[v] is the union over the heads u of
   its row of reached[u] and bit label[u] (UINT32_MAX: none), the row of an inactive v not followed (reached[v] empty);
   EINVAL also for a label at or past `columns` other than UINT32_MAX */
struct mesh_graph_closure { uint32_t vertices; size_t words; uint64_t *reached; };
static inline int mesh_graph_closure_contains(const struct mesh_graph_closure *closure, uint32_t row, uint32_t column) {
  return (int)(closure->reached[(size_t)row * closure->words + column / 64] >> (column % 64) & 1);
}
int mesh_graph_closure(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                       const void *context, const uint32_t *order, ptrdiff_t step, struct mesh_graph_closure **made);
int mesh_graph_closure_labels(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *context, size_t k),
                              const void *context, const uint32_t *order, ptrdiff_t step, const unsigned char *active,
                              uint32_t columns, const uint32_t *label, struct mesh_graph_closure **made);
void mesh_graph_closure_free(struct mesh_graph_closure *made);
/* a closure as produced, without closing again: mesh_graph_closure_save writes it into *bytes (*length of them,
   malloc'd, the caller's to free): the header {vertices (u32), 0 (u32), words (u64)}, then reached [0 .. vertices *
   words) (u64 each); mesh_graph_closure_load makes a closure of its own (mesh_graph_closure_free; its reached
   allocated as mesh_graph_closure's and copied from the bytes, which stay the caller's) from exactly those bytes,
   EINVAL where they are not one saved closure (their length, the header's zero word, or vertices * words past
   SIZE_MAX), ENOMEM */
int mesh_graph_closure_save(const struct mesh_graph_closure *closure, void **bytes, size_t *length);
int mesh_graph_closure_load(const void *bytes, size_t length, struct mesh_graph_closure **made);
/* T. H. Cormen et al., Introduction to Algorithms, 3rd ed., §24.2, DAG-SHORTEST-PATHS with max for min; S. Zhang et al.,
   HAP, EuroSys 2024, §3.2: over a retained relation's prerequisite rows (incoming: in_first/in_edge, a row's heads the
   edges' `from`; else row_first/row_edge, its heads their `to`), the vertices processed in order[0], order[step], ...
   (each vertex once, every head before its row), length[v] = 0 for a vertex with no prerequisite, else the largest
   length[head] + weight[e] over its row's edges e (weight indexed by the relation's own edges); EINVAL where the order
   is not a permutation or a head is not before its row */
struct mesh_graph_relation;
int mesh_graph_longest(const struct mesh_graph_relation *relation, int incoming, const uint32_t *order, ptrdiff_t step,
                       const double *weight, double *length);
/* F. Harary, Graph Theory, 1969 (sources and sinks); L. Lamport, CACM 21(7), 1978; M. Lam, PLDI 1988, §2.3;
   design/disjoint-pages-change-request.md C3.3 and C3.5: over a retained relation's edges e not excluded[e] (its
   mesh_graph_relation_synchronization mask) with class_of[from] == class_of[to], source[v] is 0 where such an edge enters
   v and sink[v] is 0 where one leaves v, else 1; class_of, source and sink have the relation's vertices, excluded its
   edges; returns 0 */
int mesh_graph_boundaries(const struct mesh_graph_relation *relation, const uint32_t *class_of,
                          const unsigned char *excluded, unsigned char *source, unsigned char *sink);
/* GSPMD, Y. Xu et al., arXiv 2105.04663, §3.5; G. A. Kildall, POPL 1973; J. B. Kam and J. D. Ullman, J. ACM 23(1), 1976;
   design/joint-compilation.md §3, row 1: vertex v's dims_mapping is state[first[v] .. first[v + 1]), each slot an atom
   a < atoms on mesh axis axis[a] or UINT32_MAX; edge e, edge_first[v] <= e < edge_first[v + 1], projects operand
   u = edge[e].operand into v by slots map[edge[e].forward + i], i < first[v + 1] - first[v], and v into u by slots
   map[edge[e].backward + j], j < first[u + 1] - first[u], each entry a state slot or SIZE_MAX; for l = 0 .. levels - 1,
   over the edges with level <= l, until a round updates nothing, a round being each forward projection of
   v = order[0], ..., order[vertices - 1], then each backward projection of v = order[vertices - 1], ..., order[0], a
   vertex's edges in index order both ways; a projection into w, frozen[w] zero, of candidate c (c[i] the atom at its
   slot, SIZE_MAX UINT32_MAX) writes c's atoms into w's UINT32_MAX slots unless an atom of w differs from c's atom at
   its slot or a mesh axis occurs twice in the result; returns ENOMEM or 0 */
struct mesh_graph_projection { uint32_t operand, level; size_t forward, backward; };
struct mesh_graph_sharding {
  uint32_t vertices, atoms, levels, padding;
  const uint32_t *order, *axis;
  const size_t *first, *edge_first, *map;
  const unsigned char *frozen;
  const struct mesh_graph_projection *edge;
  uint32_t *state;
};
int mesh_graph_sharding(const struct mesh_graph_sharding *sharding);
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
/* design/planner-composition.md: owned row/ordinal results; key/context and parent remain borrowed operands. */
struct mesh_graph_grouping {
  uint32_t keys;
  size_t items;
  uint32_t (*key)(const void *, size_t);
  const void *context;
  size_t *first, *order, *ordinal;
};
int mesh_graph_group(uint32_t keys, size_t items, uint32_t (*key)(const void *, size_t), const void *context,
                      struct mesh_graph_grouping **out);
void mesh_graph_group_free(struct mesh_graph_grouping *);
struct mesh_graph_tree {
  uint32_t vertices, root;
  const uint32_t *parent;
  struct mesh_graph_grouping *children;
};
int mesh_graph_tree_create(uint32_t vertices, uint32_t root, const uint32_t *parent, struct mesh_graph_tree **out);
uint32_t mesh_graph_tree_parent(const void *tree, size_t vertex);
void mesh_graph_tree_free(struct mesh_graph_tree *);
/* ExecuTorch exir/memory_planning.py update_tensor_lifetime: "A tensor's lifetime are represented by the index of the
   first and last node referring that tensor in its inputs/outputs." Use k is value value[k] at position[k]; first[v]
   and last[v] the least and greatest position of value v's uses */
int mesh_graph_lifetimes(uint32_t values, size_t uses, const uint32_t *value, const size_t *position,
                         size_t *first, size_t *last);
/* Row b contains a iff every access of a strictly precedes every access of b in ordering.
   Empty access sets are unrelated. reached has values * ceil(values/64) words. */
int mesh_graph_lifetime_order(const struct mesh_graph_relation *ordering, const uint32_t *order, uint32_t values, size_t uses,
                              const uint32_t *value, const uint32_t *vertex, uint64_t *reached);
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
/* With rate supplied, include the minimum-hop tree in the packing. rate[k] is the fraction of demand k's
   traffic carried per time unit at its share of the tightest directed edge, using the units of capacity. */
uint32_t mesh_graph_flow_rates(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share, double *rate);
/* directed[parent*n+child] is the startup in ns for a send, or uniform when directed is NULL.
   tree[k] receives the maximum root-to-leaf delay along the selected tree's directed transfers. */
struct mesh_graph_startup { double uniform; const double *directed; double *tree; };
uint32_t mesh_graph_flow_startup(uint32_t vertices, uint32_t edges, const uint32_t *from, const uint32_t *to, const double *capacity,
                         uint32_t count, const uint32_t *root, const double *up, const double *down,
                         int (*leaf)(const void *context, uint32_t d, uint32_t v), const void *context, double eps,
                         uint32_t rounds, uint32_t room, uint32_t *demand, uint32_t *parent, double *share, double *rate,
                         const struct mesh_graph_startup *startup);
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
/* design/planner-composition.md: the original weighted rows by view; query.delay is not an operand. */
struct mesh_graph_dependence;
struct mesh_graph_lag_graph {
  uint32_t vertices;
  const size_t *first;
  uint32_t (*head)(const void *, size_t);
  uint64_t (*lag)(const void *, size_t);
  const void *context;
};
struct mesh_graph_lag_result {
  const struct mesh_graph_lag_graph *graph;
  size_t queries;
  const struct mesh_graph_dependence *query;
  struct mesh_graph_grouping *sources;
  uint64_t *value;
};
struct mesh_graph_lag_matrix {
  const struct mesh_graph_lag_graph *graph;
  uint32_t sources, targets;
  const uint32_t *source, *target;
  uint64_t *value;
};
int mesh_graph_lag_prepare(uint32_t vertices, const size_t *first, uint32_t (*head)(const void *, size_t),
                            uint64_t (*lag)(const void *, size_t), const void *context,
                            struct mesh_graph_lag_graph **out);
void mesh_graph_lag_graph_free(struct mesh_graph_lag_graph *);
int mesh_graph_lag_queries(const struct mesh_graph_lag_graph *, size_t queries,
                            const struct mesh_graph_dependence *query, struct mesh_graph_lag_result **out);
uint32_t mesh_graph_query_source(const void *query, size_t index);
uint32_t mesh_graph_buffer_source(const void *buffer, size_t index);
uint32_t mesh_graph_buffer_head(const void *group, size_t index);
uint64_t mesh_graph_buffer_lag(const void *group, size_t index);
void mesh_graph_lag_result_free(struct mesh_graph_lag_result *);
int mesh_graph_lag_matrix(const struct mesh_graph_lag_graph *, uint32_t sources, const uint32_t *source,
                           uint32_t targets, const uint32_t *target, struct mesh_graph_lag_matrix **out);
void mesh_graph_lag_matrix_free(struct mesh_graph_lag_matrix *);
struct mesh_graph_event_lag_result {
  const struct mesh_graph_lag_graph *graph;
  uint32_t events;
  const size_t *first;
  const uint32_t *member;
  size_t queries;
  const struct mesh_graph_dependence *query;
  size_t *expanded_first;
  struct mesh_graph_dependence *expanded;
  struct mesh_graph_lag_result *lags;
  uint64_t *value;
};
/* compiler-emit/docs/source-analysis.md: distinct completion events A,B use max_a min_b lag(a,b);
   identical original occurrences have lag 0. Recurrence uses mesh_graph_lag_queries; its raw result is retained. */
int mesh_graph_event_lags(const struct mesh_graph_lag_graph *,uint32_t events,const size_t *first,
                          const uint32_t *member,size_t queries,const struct mesh_graph_dependence *query,
                          struct mesh_graph_event_lag_result **out);
void mesh_graph_event_lag_result_free(struct mesh_graph_event_lag_result *);

/* design/joint-compilation.md §3, row 7 */
int mesh_graph_chains(uint32_t items, const uint64_t *count, int (*before)(const void *context, uint32_t a, uint32_t b),
                      const void *context, uint64_t *chain, uint64_t *chains);
/* design/aot-compilation-change-request.md A10, "Its bound": mesh_graph_chains over items 0 .. items, item k the
   closure's vertex member[k] counted count[k], a before b where bit member[a] of reached[member[b]] is set (a path of
   the closure's rows leads from member[b] to member[a]: over rows to each vertex's predecessors, member[a] precedes
   member[b]); its chain and chains as mesh_graph_chains'. EINVAL for a member at or past the closure's vertices */
int mesh_graph_closure_chains(const struct mesh_graph_closure *closed, uint32_t items, const uint32_t *member,
                              const uint64_t *count, uint64_t *chain, uint64_t *chains);
/* design/joint-compilation.md §3, row 7 */
int mesh_graph_fat_cover(uint64_t length, uint32_t intervals, const uint64_t *start, const uint64_t *end,
                         const uint64_t *need, uint64_t *color, uint64_t *colors);
/* design/joint-compilation.md §3, row 8 */
int mesh_graph_completion_times(uint32_t vertices, const double *length, const uint32_t *resource, uint32_t resources,
                                size_t edges, const uint32_t *from, const uint32_t *to, double *completion);
/* design/unequal-shards-change-request.md items 6 and 14; S. Zhang et al., HAP, EuroSys 2024, section 5.1, (3): the
   shares B_j of `ranks` ranks that minimise, summed over `stages` stages, comm[i] times the largest B_j plus the largest
   over j of fixed[i * ranks + j] + slope[i * ranks + j] * B_j, subject to the B_j summing to 1 and 0 <= B_j <= upper[j].
   mesh_graph_shares_coupled is it over `parts` parts' shares B[v], v = p * ranks + j part p's share of rank j in [0,
   upper[j]], each part's summing to 1: comm[i * parts + p] times part p's largest share, and each of `rows` rows r of
   stage stage[r] fixed[r] plus coefficient[k] * B[variable[k]] over k in first[r] .. first[r + 1] (a row's variables
   distinct, of any parts and ranks), the stage's time its rows' largest; mesh_graph_shares is its one part, row i *
   ranks + j of stage i its variable j with slope its one coefficient. EINVAL for a stage at or past `stages` or a
   variable at or past parts * ranks */
int mesh_graph_shares(uint32_t ranks, uint32_t stages, const double *comm, const double *fixed, const double *slope,
                      const double *upper, double *share);
int mesh_graph_shares_coupled(uint32_t ranks, uint32_t parts, uint32_t stages, uint32_t rows, const double *comm,
                              const uint32_t *stage, const double *fixed, const size_t *first, const uint32_t *variable,
                              const double *coefficient, const double *upper, double *share);
/* design/joint-compilation.md §3, row 8 */
int mesh_graph_heap_order(uint32_t vertices, const double *priority, size_t edges, const uint32_t *from,
                          const uint32_t *to, uint32_t *place);
/* a view of directed rows for the order algorithms: vertices 0 .. vertices - 1; vertex v's entries k = 0 ..
   degree(context, v) - 1, entry k's head head(context, v, k), UINT32_MAX for an entry outside the view (the caller's
   restriction of retained rows: a cut, a permutation, an orientation); no rows are copied */
struct mesh_graph_rows_view {
  uint32_t vertices;
  const void *context;
  size_t (*degree)(const void *context, uint32_t v);
  uint32_t (*head)(const void *context, uint32_t v, size_t k);
};
/* mesh_graph_heap_order over a rows view: the same heap (priority, then index); EINVAL for a head out of range or a
   cycle */
int mesh_graph_heap_order_view(const struct mesh_graph_rows_view *rows, const double *priority, uint32_t *place);
int mesh_graph_relation_heap_permutation(const struct mesh_graph_relation *, const double *priority, uint32_t *place, uint32_t *order);
int mesh_graph_relation_heap_order(const struct mesh_graph_relation *, const double *priority, uint32_t *place);
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
struct mesh_graph_buffer_result {
  size_t references;
  uint32_t vertices,actors;
  const double *time,*host,*work;
  size_t edges,part_edges,queries;
  const uint32_t *from,*to;
  const uint64_t *delay;
  const struct mesh_graph_dependence *edge;
  const size_t *query;
  double lambda;
  uint64_t *bound;
  void (*release)(void *);
  void *owner;
};
/* compiler-emit/docs/source-analysis.md: query[j] names the original forward edge; bound[j] is its reverse-edge Bff.
   Input arrays remain the caller's; actors/host/work/edge retain the specialized constructor's original operands. */
int mesh_graph_buffer_queries(uint32_t vertices,const double *time,size_t edges,const uint32_t *from,
                              const uint32_t *to,const uint64_t *delay,size_t queries,const size_t *query,
                              struct mesh_graph_buffer_result **out);
int mesh_graph_buffer_bounds(uint32_t actors, const double *host, const double *work, size_t edges,
                              const struct mesh_graph_dependence *edge, struct mesh_graph_buffer_result **out);
void mesh_graph_buffer_result_retain(struct mesh_graph_buffer_result *);
void mesh_graph_buffer_result_free(struct mesh_graph_buffer_result *);
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
struct mesh_graph_access_map;
/* `access` (NULL for a strided view): the view's bytes are an access relation's (mesh_graph_access_map_create, below), its
   `view` then unused; its holder retains it for as long as the view is held */
struct mesh_graph_view { uint64_t key; struct mesh_strided view; uint32_t vertex, write; const struct mesh_graph_access_map *access; };
_Static_assert(sizeof(struct mesh_graph_view) == 56 && offsetof(struct mesh_graph_view, access) == 48,
               "mesh_graph_view: rdma/mesh_graph.py Viewed");
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
/* design/compilation-composition-change-request.md, "Complete replacement workgroup" (an upstream kernel's accesses as
   its producer relates them); S. Verdoolaege, isl, ICMS 2010 (isl 0.27: isl_pw_aff, isl_set, isl_map): an access's
   retained domain-to-byte relation, its predicate and its indirect symbols, built by the original walkers
   (torch/utils/_sympy/interp.py sympy_interp over an index; LoopBody's interpreter over its masked subblocks' masks,
   as torch/_inductor/bounds.py BoundVars runs its analysis), one isl operation per handler.
   A context: one isl_ctx and its lock, refcounted; every domain, index and access of it, and every graph query over
   them (mesh_graph_*_views and relations of access views), use it under that lock, each holder retaining it. */
struct mesh_graph_context;
int mesh_graph_context_create(struct mesh_graph_context **out);
void mesh_graph_context_retain(struct mesh_graph_context *);
void mesh_graph_context_free(struct mesh_graph_context *);
/* design/graph-isl-views.md */
struct isl_ctx;
struct isl_set;
struct isl_pw_aff;
int mesh_graph_context_lock(struct mesh_graph_context *);
int mesh_graph_context_unlock(struct mesh_graph_context *);
struct isl_ctx *mesh_graph_context_isl(const struct mesh_graph_context *);
/* A kernel's index domain, dimension k one original symbol over [lo, hi], its relation to the launch the backend's own:
   ROOT a range-tree root of an MPS kernel (torch/_inductor/codegen/simd.py IterationRangesRoot, [0, numel - 1]: a
   reduction root's whole numel, its multistage loop within one threadgroup), `axis` its position in
   active_range_trees(), the thread_position_in_grid component codegen/mps.py codegen_kernel binds it to, `reduction`
   its is_reduction (call_kernel: one threadgroup spanning it); LOOP a loop variable of one invocation (codegen/cpp.py
   LoopNest's LoopLevel var over [offset, offset + size - 1]: a CPU KernelGroup's whole call), `axis` unused; INDIRECT
   an indirect symbol (codegen/common.py CSEProxy.indirect_indexing: [0, size - 1] where its bounds are checked, else
   its var's LoopBody.bounds() value range). Refcounted; retains its context. */
enum { MESH_GRAPH_DOMAIN_ROOT, MESH_GRAPH_DOMAIN_INDIRECT, MESH_GRAPH_DOMAIN_LOOP };
struct mesh_graph_domain_dim { uint32_t kind, axis, reduction, padding; int64_t lo, hi; };
struct mesh_graph_domain;
int mesh_graph_domain_create(struct mesh_graph_context *, uint32_t dims, const struct mesh_graph_domain_dim *dim,
                             struct mesh_graph_domain **out);
void mesh_graph_domain_retain(struct mesh_graph_domain *);
void mesh_graph_domain_free(struct mesh_graph_domain *);
const struct mesh_graph_domain_dim *mesh_graph_domain_dims(const struct mesh_graph_domain *, uint32_t *dims);
/* the domain's context, borrowed for the domain's lifetime */
struct mesh_graph_context *mesh_graph_domain_context(const struct mesh_graph_domain *);
struct isl_set *mesh_graph_domain_isl(const struct mesh_graph_domain *);
/* An analysis value over a domain: an integer (isl_pw_aff) or a predicate (isl_set), as sympy_interp's and the
   LoopBody ops handler's methods of the same names make it: dimension `dim`; a constant (an integer, or `truth` a
   predicate); add, mul (one side constant), floordiv (a constant divisor: floor), modular_indexing (x, constant
   divisor, constant modulus: torch/utils/_sympy/functions.py ModularIndexing, (x floordiv d) mod m), python_mod
   (constant modulus), minimum, maximum, where (predicate, a, b); lt, le, gt, ge, eq, ne (integers to a predicate);
   and, or, not (predicates). NULL where isl has no quasi-affine value (a product of two non-constants, a non-constant
   divisor or modulus, a NULL operand); each free'd with mesh_graph_index_free. */
struct mesh_graph_index;
const struct mesh_graph_domain *mesh_graph_index_domain(const struct mesh_graph_index *);
struct isl_pw_aff *mesh_graph_index_value(const struct mesh_graph_index *);
struct isl_set *mesh_graph_index_predicate(const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_take_locked(struct mesh_graph_domain *, struct isl_pw_aff *, struct isl_set *);
struct mesh_graph_index *mesh_graph_index_dim(struct mesh_graph_domain *, uint32_t dim);
struct mesh_graph_index *mesh_graph_index_constant(struct mesh_graph_domain *, int64_t value);
struct mesh_graph_index *mesh_graph_index_truth(struct mesh_graph_domain *, int truth);
struct mesh_graph_index *mesh_graph_index_add(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_mul(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_floordiv(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_modular_indexing(const struct mesh_graph_index *x,
                                                           const struct mesh_graph_index *divisor,
                                                           const struct mesh_graph_index *modulus);
struct mesh_graph_index *mesh_graph_index_python_mod(const struct mesh_graph_index *, const struct mesh_graph_index *);
/* isl's truncated division and remainder (isl_pw_aff_tdiv_q, isl_pw_aff_tdiv_r: the quotient rounded toward zero, the
   remainder the dividend's sign; a constant divisor) */
struct mesh_graph_index *mesh_graph_index_tdiv_q(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_tdiv_r(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_minimum(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_maximum(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_where(const struct mesh_graph_index *predicate, const struct mesh_graph_index *a,
                                                const struct mesh_graph_index *b);
enum { MESH_GRAPH_INDEX_LT, MESH_GRAPH_INDEX_LE, MESH_GRAPH_INDEX_GT, MESH_GRAPH_INDEX_GE, MESH_GRAPH_INDEX_EQ,
       MESH_GRAPH_INDEX_NE };
struct mesh_graph_index *mesh_graph_index_compare(uint32_t relation, const struct mesh_graph_index *,
                                                  const struct mesh_graph_index *);
/* An integer of a typed producer's `bits`-wide type (WGSL/Tint u32, i32, u64, i64): x reduced into that type's range
   exactly (x mod 2^bits, or its two's-complement signed value where `is_signed`), as the type's arithmetic, conversion
   and bitcast define it */
struct mesh_graph_index *mesh_graph_index_wrap(const struct mesh_graph_index *x, uint32_t bits, int is_signed);
/* x's one value: 1 and `*value` it where isl proves x the same integer at every point of its domain's box (a typed
   producer's operand fixed by its constants, its overrides' values or its CONSTANT data, however computed), else 0 */
int mesh_graph_index_fixed(const struct mesh_graph_index *x, int64_t *value);
int mesh_graph_index_is_empty(const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_and(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_or(const struct mesh_graph_index *, const struct mesh_graph_index *);
struct mesh_graph_index *mesh_graph_index_not(const struct mesh_graph_index *);
void mesh_graph_index_free(struct mesh_graph_index *);
/* An access's retained relation { d -> byte : origin + scale index(d) <= byte < origin + scale index(d) + width,
   guard(d), d in its domain } (an element index: scale and width its element's bytes; a byte offset: scale 1, width
   the bytes accessed): `index` its element index (integer), or NULL where its producer's value is not
   quasi-affine (data it loaded): then its original value range [lo, hi] (BOUND, never claimed exact); `guard` its
   predicate (NULL unguarded); `flags` GUARD_DATA where its predicate is held only by its identity (data it reads, or a
   predicate its producer keeps unevaluated: it may run anywhere in its domain), UNKNOWN where its producer has no range
   of its index ([lo, hi] then the binding's bytes, stated, never exact); `guard_source` the predicate's own identity
   (its LoopBody node's or its If's, 0 none). Refcounted; retains its domain. */
enum { MESH_GRAPH_ACCESS_MAP_BOUND = 1, MESH_GRAPH_ACCESS_MAP_GUARD_DATA = 2, MESH_GRAPH_ACCESS_MAP_UNKNOWN = 4 };
int mesh_graph_access_map_create(const struct mesh_graph_domain *domain, const struct mesh_graph_index *index, int64_t lo,
                                 int64_t hi, const struct mesh_graph_index *guard, uint32_t flags,
                                 uint64_t guard_source, uint64_t origin, uint64_t scale, uint64_t width,
                                 struct mesh_graph_access_map **out);
int mesh_graph_access_map_union(const struct mesh_graph_access_map *, const struct mesh_graph_access_map *,
                                 struct mesh_graph_access_map **out);
void mesh_graph_access_map_retain(const struct mesh_graph_access_map *);
void mesh_graph_access_map_free(const struct mesh_graph_access_map *);
/* its flags, guard identity and domain, and its bytes' least and greatest (an empty relation: 1 and 0) */
uint32_t mesh_graph_access_map_flags(const struct mesh_graph_access_map *, uint64_t *guard_source);
const struct mesh_graph_domain *mesh_graph_access_map_domain(const struct mesh_graph_access_map *);
/* its identity: made once with the relation (64 random bits), the same in every saved and loaded copy of it, a new one
   for each relation made of it (mesh_graph_access_map_placed) */
uint64_t mesh_graph_access_map_identity(const struct mesh_graph_access_map *);
void mesh_graph_access_map_span(const struct mesh_graph_access_map *, int64_t *least, int64_t *greatest);
/* The same relation's bytes in [start, end) moved by `shift` (a placement's page or copy: new physical bytes of one
   relation), its flags kept; *out NULL where none are */
int mesh_graph_access_map_placed(const struct mesh_graph_access_map *, int64_t start, int64_t end, int64_t shift,
                             struct mesh_graph_access_map **out);
/* The partitions of a received range each threadgroup g = (g_0, g_1, g_2) of a configured launch reaches: the
   relation's domain restricted by the launch's own thread relation (Metal dispatchThreads: thread_position_in_grid =
   g threads_per_threadgroup + thread_position_in_threadgroup), each ROOT dimension that is no reduction to its
   threadgroup's positions on its axis, [g_axis group[axis], + group[axis]) (group[axis] 0: the axis whole, one
   invocation's), every other dimension whole (a reduction root, a LOOP variable, an INDIRECT symbol), 0 <= g < groups, its bytes mapped to partition p of [start + p page, + page), 0 <= p < partitions (isl:
   the image of the relation under that map, its parameters each threadgroup's); for threadgroup w = g_0 + groups[0]
   (g_1 + groups[1] g_2) its partitions, ascending, partition[first[w] .. first[w + 1]) (*partition malloc'd, the
   caller's; first[groups' product + 1] the caller's). A BOUND or GUARD_DATA relation's are its own data-dependent
   bound, as its flags state */
/* Accesses grouped by launch coordinates. A consumer may share the producer's tile completion when
   read availability is included groupwise, and every byte hazard connects equal group coordinates. */
int mesh_graph_access_groups(const uint32_t threads[3], const uint32_t groups[3],
                             size_t producers, const struct mesh_graph_access_map *const *producer,
                             size_t consumers, const struct mesh_graph_access_map *const *consumer,
                             int subset, int *proven);
struct mesh_graph_access_pair {
  const struct mesh_graph_access_map *producer, *consumer;
};
int mesh_graph_group_incidence(const uint32_t producer_threads[3], const uint32_t producer_groups[3],
                               const uint32_t consumer_threads[3], const uint32_t consumer_groups[3],
                               size_t pairs, const struct mesh_graph_access_pair *, size_t *first, uint32_t **members);
/* The phase fields of planned extents whose byte intervals each threadgroup reads. */
int mesh_graph_access_map_fields(const struct mesh_graph_access_map *, const uint32_t group[3], const uint32_t groups[3],
                                 uint64_t origin, size_t count, const struct mesh_graph_access *extents,
                                 size_t *first, uint32_t **fields);
int mesh_graph_access_map_partitions(const struct mesh_graph_access_map *, const uint32_t group[3],
                                     const uint32_t groups[3], int64_t start, uint64_t page, uint32_t partitions,
                                     size_t *first, uint32_t **partition);
/* Each borrowed map's launch-domain image in pages relative to its own start; group and groups use the same
   restriction as access_map_partitions, existentially over all executed groups. Whole explicitly names all pages
   [0, partitions). The wanted union is covered by the covering union. Unknown/bounded/data-guarded, indirectly
   indexed or unknown-launch covering maps supply no guaranteed pages; unknown-launch wanted maps retain their
   full domain image. Different map domains may share one context. A different context gives no proof.
   Both sides must describe the same access stage and page coordinate system; execution/epoch order is the caller's. */
struct mesh_graph_access_pages {
  const struct mesh_graph_access_map *map; int64_t start; uint32_t whole;
  const uint32_t *group, *groups;
};
int mesh_graph_access_pages_cover(uint64_t page, uint64_t partitions,
                                  size_t covers, const struct mesh_graph_access_pages *cover,
                                  size_t wanted, const struct mesh_graph_access_pages *want, int *covered);
/* isl's own text of the relation and its guard (isl_map_to_str, isl_set_to_str) with its domain, flags, guard
   identity and span: mesh_graph_access_map_save into *bytes (malloc'd, the caller's); mesh_graph_access_map_load the
   same relation, its domain and itself made in `context` */
int mesh_graph_access_map_save(const struct mesh_graph_access_map *, void **bytes, size_t *length);
int mesh_graph_access_map_load(struct mesh_graph_context *context, const void *bytes, size_t length,
                               struct mesh_graph_access_map **out);
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
/* mesh_graph_levels over a rows view: level[v] the longest path to v; EDEADLK for an entry whose head is not after its
   vertex, EINVAL for a head out of range */
int mesh_graph_levels_view(const struct mesh_graph_rows_view *rows, uint32_t *level);
/* design/synchronization-signature-change-request.md items 1, 2, 4 and 6; S. S. Bhattacharyya, S. Sriram and E. A. Lee,
   IEEE Trans. Signal Processing 45(6), 1997, §4.1, §5.1 and §7, Fig. 4; E. A. Lee and D. G. Messerschmitt, IEEE Trans.
   Computers C-36(1), 1987; M. Lam, PLDI 1988; L. Lamport, CACM 21(7), 1978 */
#define MESH_GRAPH_NONE UINT32_MAX
enum { MESH_GRAPH_QUEUE, MESH_GRAPH_HOST, MESH_GRAPH_BRIDGE, MESH_GRAPH_KERNEL, MESH_GRAPH_KINDS };
enum { MESH_GRAPH_SUBMIT, MESH_GRAPH_QUEUE_SIGNAL, MESH_GRAPH_QUEUE_WAIT, MESH_GRAPH_FENCE_UPDATE, MESH_GRAPH_FENCE_WAIT,
       MESH_GRAPH_PASS_BARRIER, MESH_GRAPH_QUEUE_BARRIER, MESH_GRAPH_HOST_WAIT, MESH_GRAPH_HOST_NOTIFY,
       MESH_GRAPH_BRIDGE_START, MESH_GRAPH_THREADGROUP_BARRIER, MESH_GRAPH_SIMDGROUP_BARRIER, MESH_GRAPH_ATOMIC,
       MESH_GRAPH_PRODUCER_BARRIER, MESH_GRAPH_BRIDGE_RAISE, MESH_GRAPH_INITIAL, MESH_GRAPH_HOST_PUBLISH,
       MESH_GRAPH_KERNEL_FENCE, MESH_GRAPH_HOST_SIGNAL, MESH_GRAPH_BRIDGE_READ, MESH_GRAPH_PRIMITIVES };
/* MESH_GRAPH_SCOPED: a wait answered by a completion scope's publication event (mesh_graph_signature_scopes), not by
   an operation: no task-graph edge is projected from it */
enum { MESH_GRAPH_IMAGE, MESH_GRAPH_STRONGER, MESH_GRAPH_FALSE, MESH_GRAPH_UNMATCHED, MESH_GRAPH_SATISFIED,
       MESH_GRAPH_TRANSFORMED, MESH_GRAPH_SCOPED, MESH_GRAPH_STATES };
struct mesh_graph_processor { uint32_t kind, index; };
/* one program's identities across its producers: a processor's `index` is thread * MESH_GRAPH_NODES + node (the host's
   program thread 0 and its event listener 1; a queue's ordinal; the bridge 0), node its map index; a plan count word
   is object MESH_GRAPH_PLAN_COUNT + its added edge, a compiled function's submissions MESH_GRAPH_PLAN_SUBMISSION + its
   index in mesh_plan's `functions`, each submission's value the plan's executions after it (mesh_transfers_start) */
#define MESH_GRAPH_NODES 64u
#define MESH_GRAPH_PLAN_COUNT UINT32_C(0x80000000)
#define MESH_GRAPH_PLAN_SUBMISSION UINT32_C(0xC0000000)
#define MESH_GRAPH_LOWERED UINT32_C(0x40000000)
/* a MESH_GRAPH_BRIDGE_READ's object: MESH_GRAPH_PLAN_SUPPLIED + the representative transfer (its owner call's row) of
   the release whose supplied word the bridge's SEND hop reads (rdma/mesh-collective.c plan_supplied), its owner call
   its vertex's task's `label` */
#define MESH_GRAPH_PLAN_SUPPLIED UINT32_C(0x20000000)
struct mesh_graph_task { uint32_t processor, place, label, copy; };
struct mesh_graph_dependence { uint32_t from, to; uint64_t delay; };
struct mesh_graph_chain_result {
  uint32_t vertices;
  const uint32_t *chain;
  size_t edges;
  struct mesh_graph_dependence *edge;
};
int mesh_graph_resource_cycles(uint32_t vertices, const uint32_t *order, uint32_t resources,
                               const uint32_t *resource, int singletons, struct mesh_graph_dependence *edge, size_t *edges);
int mesh_graph_chain_pairs(uint32_t vertices, const uint32_t *chain, struct mesh_graph_chain_result **out);
void mesh_graph_chain_result_free(struct mesh_graph_chain_result *);
/* design/planner-composition.md: original byte touches and value/relay operands; derived complete cover sets. */
struct mesh_graph_touch { uint64_t start, end; uint32_t value, member; };
struct mesh_graph_cover { const uint32_t *member; uint32_t members, value, block; };
struct mesh_graph_cover_result {
  uint64_t quantum;
  uint32_t values, blocks, extras, sets;
  size_t touches, covers, listed, held_count;
  const uint32_t *block_first;
  uint64_t (*bytes)(const void *, uint32_t);
  const void *context;
  const struct mesh_graph_touch *touch;
  uint32_t (*extra_member)(const void *, uint32_t);
  uint64_t (*extra_blocks)(const void *, uint32_t);
  const void *extra_context;
  const struct mesh_graph_touch **order;
  struct mesh_graph_cover *cover;
  uint32_t *held, *first, *member, *set_blocks, *block_set, *block_element, *extra_set, *uncovered;
};
int mesh_graph_cover_sets(uint64_t quantum, uint32_t values, uint32_t blocks, const uint32_t *block_first,
                           uint64_t (*bytes)(const void *, uint32_t), const void *context, size_t touches,
                           const struct mesh_graph_touch *touch, uint32_t extras,
                           uint32_t (*extra_member)(const void *, uint32_t),
                           uint64_t (*extra_blocks)(const void *, uint32_t), const void *extra_context,
                           struct mesh_graph_cover_result **out);
void mesh_graph_cover_result_free(struct mesh_graph_cover_result *);
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
struct mesh_graph_witness { size_t record; uint32_t reverse, known; };
_Static_assert(sizeof(struct mesh_graph_witness) == 16, "mesh_graph_witness");
/* a record's deciding original (mesh_graph_relation_decided): the retained relation and its witness, oriented as the
   record is */
struct mesh_graph_origin { const struct mesh_graph_relation *relation; struct mesh_graph_witness witness; };
_Static_assert(sizeof(struct mesh_graph_origin) == 24, "mesh_graph_origin");
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
  uint32_t query;
  const struct mesh_graph_origin *origin;
};
/* The given dependency at index k is conflict k. Vertex IDs are independent of execution order. */
int mesh_graph_relation_edges(uint32_t vertices, size_t edges, const uint32_t *from, const uint32_t *to,
                              struct mesh_graph_relation **out);
void mesh_graph_relation_retain(struct mesh_graph_relation *);
void mesh_graph_relation_free(struct mesh_graph_relation *);
int mesh_graph_relation_create(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t conflicts,
                               const struct mesh_graph_conflict *conflict, void (*release)(void *), void *owner,
                               struct mesh_graph_relation **out);
const struct mesh_graph_relation_data *mesh_graph_relation_data(const struct mesh_graph_relation *);
/* Incoming closure from the original incoming-row search's postorder; freed by mesh_graph_closure_free. */
int mesh_graph_relation_ancestors(const struct mesh_graph_relation *, struct mesh_graph_closure **made);
int mesh_graph_relation_quotient(struct mesh_graph_relation *parent, uint32_t classes, const uint32_t *class_of,
                                 void (*release)(void *), void *owner, struct mesh_graph_relation **out);
int mesh_graph_relation_contract(struct mesh_graph_relation *parent, const unsigned char *kept, void (*release)(void *),
                                 void *owner, struct mesh_graph_relation **out);
/* M. Lam, PLDI 1988, §2.3 (modulo variable expansion, the unrolled copies' inverse): `parent` over `lag` iterations of
   `count` vertices folded onto its last iteration, a derived relation over `count` vertices: each parent edge into
   the last iteration, from x to (lag - 1) count + y, as {x % count, y, its delay + lag - 1 - x / count}, in the
   parent's edge order, its one source that parent edge (its members that edge's); its operands `count` and `lag`,
   retained and archived (mesh_graph_relation_newest_operands: 0 and them for such a relation, ENOENT for any other).
   EINVAL where `count` or `lag` is 0 or the parent has more than lag x count vertices */
int mesh_graph_relation_newest(struct mesh_graph_relation *parent, uint32_t count, uint32_t lag,
                               struct mesh_graph_relation **out);
int mesh_graph_relation_newest_operands(const struct mesh_graph_relation *, uint32_t *count, uint32_t *lag);
int mesh_graph_relation_synchronization(struct mesh_graph_relation *, const unsigned char **redundant, size_t *kept);
/* design/compilation-composition-change-request.md items 6.4 and 7: a root made by mesh_graph_relation_views is the
   one mesh_graph_dependences_views (self 0, query DIRECTED) or mesh_graph_conflicts_views (self 1, query SELF) solve
   over `view` and the supplied edges, its conflict records owned by the relation, `owner` (views, pairs, supplied)
   adopted as mesh_graph_relation_create's; a root of mesh_graph_relation_create has query NONE; a derived relation
   has its root's. mesh_graph_relation_witness answers the orientation from_view -> to_view of two of the root's
   original views from that query alone: known 0 where it did not decide the pair (NONE, or a pair of one vertex under
   DIRECTED); else record the original conflict record of the pair (reverse 1 where that record orients it
   to_view -> from_view) or SIZE_MAX, no hazard. GIVEN records are no evidence. mesh_graph_relation_cause is the
   witness's hazard in the asked orientation, view_conflicts' own: key and given the record's, from_view and to_view
   as asked; ENODATA where not known, ENOENT where there is none */
enum { MESH_GRAPH_QUERY_NONE, MESH_GRAPH_QUERY_DIRECTED, MESH_GRAPH_QUERY_SELF };
int mesh_graph_relation_views(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t supplied,
                              const uint32_t *from, const uint32_t *to, int self, void (*release)(void *), void *owner,
                              struct mesh_graph_relation **out);
int mesh_graph_relation_witness(const struct mesh_graph_relation *, size_t from_view, size_t to_view,
                                struct mesh_graph_witness *out);
int mesh_graph_relation_cause(const struct mesh_graph_relation *, const struct mesh_graph_witness *,
                              struct mesh_graph_cause *out);
int mesh_graph_relation_edge_cause(const struct mesh_graph_relation *, size_t edge, uint32_t *kind);
/* design/compilation-composition-change-request.md items 6.4 and 7: mesh_graph_relation_views (self and query as
   there) whose pair test asks `decide` first. For each candidate pair of distinct views a and b (one key, hulls that
   meet, not both reads, not both write 2, both nonempty), decide(context, a, b, &d) either answers it from a retained
   relation (d.relation, and d.witness that relation's mesh_graph_relation_witness for the pair oriented a -> b:
   known 1, record its original record or SIZE_MAX for its established absence), or leaves it (witness.known 0) to be
   solved here, once. A nonzero return ends the query with that status. The records are the solver's own, as
   mesh_graph_relation_views makes them. data.origin[k] is record k's deciding relation and witness, `reverse`
   relative to record k's own from_view -> to_view, and {NULL, {SIZE_MAX, 0, 0}} for a pair solved here or a GIVEN
   record. d.map is the deciding relation's map: its views' identities in the caller's namespace (borrowed under
   `owner`); `map` is this query's own (view v's identity map[v], borrowed under `owner`, NULL for none).
   mesh_graph_relation_deciders gives every (relation, map) input that answered one of the root's pairs, hazard or
   absence, once in the order first answered (*relation, retained by the result until its last reference), its map
   (*maps) and the root's own map (*map): the provenance of each answer, an absence included, with no table of
   pairs. Derived relations share their root's origin and deciders. 0 deciders and NULL maps for a relation not made
   here */
struct mesh_graph_decided { struct mesh_graph_relation *relation; const uint32_t *map; struct mesh_graph_witness witness; };
int mesh_graph_relation_decided(uint32_t vertices, size_t views, const struct mesh_graph_view *view, size_t supplied,
                                const uint32_t *from, const uint32_t *to, int self, const uint32_t *map,
                                int (*decide)(void *context, size_t a, size_t b, struct mesh_graph_decided *out),
                                void *context, void (*release)(void *), void *owner, struct mesh_graph_relation **out);
size_t mesh_graph_relation_deciders(const struct mesh_graph_relation *, struct mesh_graph_relation *const **relation,
                                    const uint32_t *const **maps, const uint32_t **map);
/* design/compilation-composition-change-request.md item 7, "fusion may discharge internal edges but cannot erase
   required ordering through outside work": whether vertices member[0 .. members) of `r` can be made one. *convex is 0
   where a path of r's outgoing rows leaves the members and returns to them, and 1 otherwise. A path among the members
   is discharged, as is an edge from a member to one of copy[0 .. copies), a vertex holding its own copy of the
   members' work. One mesh_graph_search of the rows, from the heads of the members' rows that are neither members nor
   copies. EINVAL for a member or copy out of range */
int mesh_graph_relation_convex(const struct mesh_graph_relation *, size_t members, const uint32_t *member, size_t copies,
                               const uint32_t *copy, int *convex);
/* design/compilation-composition-change-request.md items 3, 6.4 and 7: the complete retained result, saved and loaded
   without solving, deriving or projecting again. mesh_graph_relation_save writes into *bytes (*length of them, the
   caller's, released with free) the relation and every relation it references, each once (its parent, a root's
   deciding relations, theirs), in mesh_graph_search's postorder from it: each root's original views with their pairs,
   its conflict records, witness index, origin records (each record's deciding relation and witness), deciders, its
   own map and its deciders' maps (mesh_graph_relation_deciders);
   each relation's edges, members, outgoing and incoming rows and query; a derived relation's sources and operand
   (class_of, kept, or newest count and lag); and its synchronization mask and kept count where it was computed.
   mesh_graph_relation_load returns the same relations, the same references between them, with every array a borrowed
   span of `bytes` (8-byte aligned, EINVAL otherwise), the maps included, except each root's view headers (pairs
   relocated into `bytes`), origin records (relations relocated), deciders (retained) and the deciders' map pointers. On success every loaded root shares `owner`, adopted
   as mesh_graph_relation_create does: release(owner) once the last of them is freed, `bytes` borrowed until then. On
   failure the caller keeps `owner` and nothing is kept. EINVAL for an archive whose spans or indices are out of
   bounds. An access view's relation (struct mesh_graph_access_map) is saved as isl's own text and loaded into
   `context`, the loading compilation's one analysis context (retained by the loaded relations; EINVAL where the
   archive holds one and `context` is NULL), so every relation loaded or made in that compilation composes with it */
int mesh_graph_relation_save(const struct mesh_graph_relation *, void **bytes, size_t *length);
int mesh_graph_relation_load(struct mesh_graph_context *context, const void *bytes, size_t length,
                             void (*release)(void *), void *owner, struct mesh_graph_relation **out);
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
/* design/synchronization-signature-change-request.md items 1, 4 and 6: released values' completion scopes as the
   native compiler made them, in these shared canonical records (the native compiler's own completion and writer
   vectors are of these types, native fa9f4a8a), borrowed under their owner: completion c, counter `counter`
   of value `value` completed `total` times an execution by writers writer[c.writer .. c.writer + c.writers) (each its
   dispatch, the dispatch incrementing for it, its metal_ir_laters index `later`, MESH_GRAPH_NONE for a host or
   unsplit writer, whose readiness the owner holds, and its completions `count`), publishing each of its releases
   release[release_first[c] .. release_first[c + 1]) (its owner call and representative transfer) once, after the
   last of them, whichever writer's that is */
struct mesh_graph_completion { uint32_t value, counter, writer, writers; uint64_t total; };
struct mesh_graph_completion_writer { uint32_t dispatch, completion, later, padding; uint64_t count; };
struct mesh_graph_release { uint32_t owner, transfer; };
_Static_assert(sizeof(struct mesh_graph_completion) == 24 && sizeof(struct mesh_graph_completion_writer) == 24 &&
               sizeof(struct mesh_graph_release) == 8, "rdma/mesh_graph.py Completion, CompletionWriter, Release");
/* a scope row's native source: `row` its original native artifact row (the loaded artifact's
   nativePreparation.calls[row].program), `node` the native signature node that emitted it (MetalProgramSignatureNode's
   node argument: its bridge processor's index, every writer of the row on it) and `identity` that row's existing
   native STRUCTURAL identity (metal_ir_identity's declaration and readiness operands; not a digest of its numerical
   code or binary), the reference by which a reader re-finds the row in its own artifact */
struct mesh_graph_scope_source { uint32_t row, node; unsigned char identity[32]; };
_Static_assert(sizeof(struct mesh_graph_scope_source) == 40, "rdma/mesh_graph.py ScopeSource");
/* one row's arrays and its source (NULL: a row of no native source, which answers no read), all borrowed */
struct mesh_graph_scopes_data { size_t completions, writers, releases; const struct mesh_graph_completion *completion;
                                const struct mesh_graph_completion_writer *writer; const size_t *release_first;
                                const struct mesh_graph_release *release;
                                const struct mesh_graph_scope_source *source; };
_Static_assert(sizeof(struct mesh_graph_scopes_data) == 64, "rdma/mesh_graph.py ScopesData");
/* one row's scopes (struct mesh_graph_scopes_data: one native row Program's own vectors, its counts and borrowed
   arrays). mesh_graph_scopes_create: reference counted as struct mesh_graph_relation is, over rows row[0 .. rows), the
   descriptors and their arrays borrowed (no copy), adopting `owner` on success (release_owner(owner) with the last
   reference; NULL: the caller keeps them alive; on failure the caller keeps it), retained by every signature holding
   it; mesh_graph_scopes_rows its descriptors. EINVAL for a row whose writer range or release range is out of bounds,
   or whose writers' counts do not sum to its total */
struct mesh_graph_scopes;
int mesh_graph_scopes_create(size_t rows, const struct mesh_graph_scopes_data *row, void (*release_owner)(void *),
                             void *owner, struct mesh_graph_scopes **out);
void mesh_graph_scopes_retain(struct mesh_graph_scopes *);
void mesh_graph_scopes_free(struct mesh_graph_scopes *);
const struct mesh_graph_scopes_data *mesh_graph_scopes_rows(const struct mesh_graph_scopes *, size_t *rows);
/* the scopes' original native Programs: mesh_graph_scopes_bind sets source[0 .. rows) (one opaque pointer per scope
   row, NULL for none; `rows` the scopes' row count, EINVAL otherwise), the array and what it points to borrowed under
   `owner`, adopted on success (release_owner(owner) once the binding is replaced or the scopes freed; NULL: the caller
   keeps them alive; on failure the caller keeps it, the earlier binding kept). The graph neither reads nor validates a
   pointer: the native resolver matches each row's mesh_graph_scope_source against its Program first. The binding is
   apart from the descriptors' owner, which stays retained. mesh_graph_scopes_sources: the bound array, NULL unbound
   (scopes read from a record or archive are unbound: their sources are references until bound) */
int mesh_graph_scopes_bind(struct mesh_graph_scopes *, size_t rows, const void *const *source,
                           void (*release_owner)(void *), void *owner);
const void *const *mesh_graph_scopes_sources(const struct mesh_graph_scopes *);
/* one realized occurrence of row `row`'s completion `completion`: `execution` the block execution t (its
   MetalSitesEmit operations' execution), `generation` the writer invocation e = t every + ordinal of its row, `copy`
   e modulo its counter placement's copies (the counter reset to zero for it; no runtime ring generation word is
   kept); its writers' sites site[site .. site + writers) and its releases' publication events [event, event +
   releases), event k's published value published[k] (its release owner's execution e + 1). A site is the
   writer's actual completion sites in that occurrence: `first` the task of the one completing its ready groups (a host
   writer's one site), `later` its later pass's (MESH_GRAPH_NONE: none), `first_site` and `later_site` their native
   sites. A publication event is an index of `published`, disjoint from operations and tasks: it is no operation and no
   processor's, and occurs once after its occurrence's `total` completions */
struct mesh_graph_occurrence { uint32_t completion, site, event, row; uint64_t execution, generation, copy; };
struct mesh_graph_site { uint32_t first, later, first_site, later_site; };
/* a MESH_GRAPH_BRIDGE_READ operation `operation` answered by publication event `event`: the event's row's source node
   the read's processor's index, its release owner the read's vertex's task's label, its representative transfer the
   read's object less MESH_GRAPH_PLAN_SUPPLIED, its value the read's (an event of a row without source answers none) */
struct mesh_graph_answered { uint32_t event, operation; };
_Static_assert(sizeof(struct mesh_graph_occurrence) == 40 && sizeof(struct mesh_graph_site) == 16 &&
               sizeof(struct mesh_graph_answered) == 8, "rdma/mesh_graph.py Occurrence, Site, Answered");
struct mesh_graph_signature { uint32_t processors, tasks; size_t operations, dependences, synchronized, uncovered,
                              redundant, unwaited, states[MESH_GRAPH_STATES];
                              struct mesh_graph_processor *processor; struct mesh_graph_task *task;
                              struct mesh_graph_operation *operation; struct mesh_graph_dependence *dependence;
                              struct mesh_graph_synchronized *edge; uint64_t *position, *count, *least;
                              size_t *previous, *waited, *first, *order; uint64_t *unknown; uint32_t *alias;
                              unsigned char digest[32]; size_t events; struct mesh_graph_event *event;
                              size_t spans; struct mesh_graph_span *span; size_t repeats;
                              struct mesh_graph_repeated *repeated; size_t proofs; struct mesh_graph_proof *proof;
                              struct mesh_graph_scopes *scopes; struct mesh_graph_realized *realized;
                              size_t occurrences, sites, publications, answers;
                              const struct mesh_graph_occurrence *occurrence; const struct mesh_graph_site *site;
                              const uint64_t *published; struct mesh_graph_answered *answer;
                              struct mesh_graph_site *projected; struct mesh_graph_archived *archived; };
_Static_assert(sizeof(struct mesh_graph_signature) == 416, "rdma/mesh_graph.py Signature");
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
/* a built signature's completion scopes (retained) and their realized occurrences, sites and publication events, the
   producer's own arrays borrowed (no copy) and its `owner` adopted on success (release_owner(owner) once the signature,
   and every signature composed from it, is freed; NULL: the caller keeps them alive), replacing any earlier ones (their
   references released, their SCOPED pairs UNMATCHED again before the new answers), as one transaction: on failure the
   signature is as it was (its earlier scopes, arrays, states and digest), the caller keeps `owner` and its arrays, and
   no release_owner call is made; every MESH_GRAPH_BRIDGE_READ whose (node, owner, representative transfer, value) is
   one event's answered by it (struct mesh_graph_answered; its pair MESH_GRAPH_SCOPED); the counts and digest again. No pair, edge or least delay of the
   task graph changes: an event has no task, so no obligation is covered by it. `realized` the adopted arrays' holder
   (reference counted); `projected` a composed signature's sites, their tasks moved into its task numbering (NULL
   otherwise). EINVAL for an occurrence's row, completion, sites or events out of range, or a site's task out of
   range */
struct mesh_graph_realized;
int mesh_graph_signature_scopes(struct mesh_graph_signature *, struct mesh_graph_scopes *, size_t occurrences,
                                const struct mesh_graph_occurrence *, size_t sites, const struct mesh_graph_site *,
                                size_t publications, const uint64_t *published, void (*release_owner)(void *),
                                void *owner);
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
/* mesh_graph_archive: the solved signature in full (its record, then its derived arrays and its answered reads: struct
   mesh_graph_answered, their count in the archive's header, with its counts and states). mesh_graph_archive_read: that
   result exactly as solved, nothing derived, paired, answered or digested again: every array of the signature, primary,
   derived, answered, its program's and its scopes' (each row's mesh_graph_scope_source with it; no pointer is saved),
   a view of `bytes` (8-byte aligned, EINVAL otherwise), its counts, states and digest the archive's. `archived` holds
   them: `owner` adopted on success, scoped or not (release_owner(owner) once the signature and every signature holding
   its scopes or realized arrays are freed; NULL: the caller keeps `bytes` alive); on failure the caller keeps it. A
   signature's views are never written or freed: a mutation (mesh_graph_signature_scopes, _spans, _program,
   mesh_graph_resynchronized) makes its own copy of only the array it changes, and mesh_graph_signature_free frees only
   arrays outside `bytes`. Restored scopes are unbound, their rows' sources explicit references (artifact row, node,
   structural identity) and no resolved native Program until the native resolver binds the owning artifact's
   (mesh_graph_scopes_bind). EINVAL for an archive of another version (its magic: no older archive is re-solved).
   mesh_graph_record_read, the tool's conversion from a record, is apart: it copies the record's primary arrays and
   derives the rest as mesh_graph_signature does (pairs, counts, answers, digest), its scopes' views borrowed of the
   record, which the caller keeps alive */
struct mesh_graph_archived;
int mesh_graph_archive_read(const void *bytes, size_t length, void (*release_owner)(void *), void *owner,
                            struct mesh_graph_signature *);
/* design/synchronization-signature-change-request.md item 1; L. Lamport, CACM 21(7), 1978: one program's signature from
   two producers' (a: the plan's, b: the replay's). Processors are united by identity, a's tasks and dependences come
   first and b's after, each producer's pairs keep their states, witnesses and dependences, and a native operation
   both list (BRIDGE_START, same processor identity, object and value) is b's, its `alias` a's task, with a's operations
   on that processor before it in a's order placed before it. Both producers' completion scopes and realized
   occurrences are kept, each distinct occurrence and publication event once: a's rows, sites, events and tasks first,
   b's after them (b's row r a's rows + r, its own row for the same scopes; site k a's sites + k; event k a's
   publications + k; task t a's tasks + t), each original scopes, source binding and realized holder retained (each
   owner kept) and only the composed descriptors (their sources), bound source pointers, occurrences, sites and
   published values made (mesh_graph_signature_scopes' projected);
   every BRIDGE_READ answered over all events, no task edge added. EOVERFLOW where a composed row, site or event index
   exceeds uint32_t */
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
/* compiler-emit/docs/physical-cuts.md */
int mesh_graph_cut_join(uint64_t intervals, const uint64_t *cut, uint32_t spans,
                        const struct mesh_graph_interval *span, uint64_t previous, const uint64_t *previous_cut,
                        uint32_t *route, uint64_t *count, uint64_t *first, uint64_t *last);
/* GaijinEntertainment/DagorEngine 2baf1947, prog/gameLibs/render/daFrameGraph/backend/resourceScheduling/heapLayout.h:19
   and heapLayout.cpp:13-23, do_arcs_overlap */
int mesh_graph_arcs_overlap(uint64_t a_start, uint64_t a_end, uint64_t b_start, uint64_t b_end);
/* TensorFlow Lite 6cf18d88, tensorflow/lite/delegates/gpu/common/memory_management/greedy_by_size_assignment.h:43-45,
   GreedyBySizeAssignment; design/torch-capture-change-request.md item 6.  `pinned` (NULL: none), a bitmap of records
   (bit r % 64 of word r / 64) whose offset[r] is given and kept: they are assigned first, and the others placed around
   those they intersect */
int mesh_graph_greedy_by_size(size_t records, const uint64_t *size, const uint64_t *pinned,
                              int (*intersect)(const void *context, size_t a, size_t b),
                              const void *context, uint64_t align, uint64_t *offset, uint64_t *total);

void mesh_graph_row_visit(const uint32_t *row, uint32_t count, void (*visit)(void *context, uint32_t k), void *context);
/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §5.2; B. R. Rau, MICRO 1994;
   design/torch-source-lowering-change-request.md item 3: over row `row` of rows (first, order), each edge e = order[q]
   not redundant[e] (NULL: none) whose tail(context, e), a vertex of `lag` unrolled iterations of `count`, is
   delay = lag - 1 - tail / count iterations before, where t >= delay: its source tail % count, its slot
   (t - delay) % lag and (edge non-NULL) e, in row order; returns the count made. mesh_graph_delayed is it over a flat
   tail array; mesh_graph_relation_delayed over a relation's incoming rows from vertex `offset` (in_first + offset,
   in_edge, the tail edge[e].from: mesh_graph_relation_tail), e the relation's edge */
size_t mesh_graph_delayed_rows(const size_t *first, const size_t *order, uint32_t (*tail)(const void *context, size_t e),
                               const void *context, const unsigned char *redundant, uint32_t row, uint32_t count,
                               uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot, size_t *edge);
size_t mesh_graph_delayed(const size_t *first, const size_t *order, const uint32_t *tail, uint32_t row, uint32_t count,
                          uint32_t lag, uint64_t t, uint32_t *source, uint32_t *slot, size_t *edge);
uint32_t mesh_graph_relation_tail(const void *context, size_t e);
size_t mesh_graph_relation_delayed(const struct mesh_graph_relation_data *relation, const unsigned char *redundant,
                                   uint32_t offset, uint32_t row, uint32_t count, uint32_t lag, uint64_t t,
                                   uint32_t *source, uint32_t *slot, size_t *edge);
/* design/compilation-composition-change-request.md item 6: the producer execution e a consumer's execution
   e_b = b every_b + ordinal_b reads, e = (e_b / every) producer + ordinal in 64 bits, every = every_b, producer the
   producing function's every_f, ordinal the producing call's ordinal_f; {1, 1, 0} the consumer's own (e = e_b). Its
   readiness: mesh-metal.h struct mesh_metal_readiness, compiled by metal-microbench metal_transport.mm */
struct mesh_graph_generation { uint32_t every, producer, ordinal; };

#ifdef __cplusplus
}
#endif
#endif
