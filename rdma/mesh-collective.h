#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"
#include "mesh-graph.h"

/* design/page-queue-change-request.md, "What the collective decides, and what the bridge decides" and "The functions";
   design/disjoint-pages-change-request.md, Appendix C, C3.1 to C3.7, and D.3.3 and D.3.6;
   design/plan-to-bridge-change-request.md R1 to R5 and R8;
   design/aot-compilation-change-request.md A2, A3, A6 to A10 and appendix items 1, 2, 5, 6 and 7;
   M. Lam, Software pipelining, PLDI 1988; B. R. Rau, Iterative modulo scheduling, MICRO 1994;
   L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical Memo 33, §2.2;
   J. Gibbons, J. Functional Programming, 1996; O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007;
   O. Sinnen and L. A. Sousa, Communication contention in task scheduling, IEEE TPDS 16(6) 2005, Definition 4;
   M. Cowan et al., MSCCLang: Microsoft Collective Communication Language, ASPLOS 2023, §4.2 and §5.2;
   C.-Y. Chen, arXiv 2205.02474, 2022, (14), (14a), (1b) and (1c);
   L. A. Hall, A. S. Schulz, D. B. Shmoys and J. Wein, Math. Oper. Res. 22(3) 1997, (5);
   F. B. Schneider, ACM Comput. Surv. 22(4), 1990; R. Frostig, M. J. Johnson and C. Leary, SysML 2018;
   NVIDIA, CUDA Programming Guide §4.2.2; Pathways §3; R. Ross, N. Miller and W. D. Gropp, EuroPVM/MPI 2003;
   J. Wu, P. Wyckoff and D. K. Panda, IPDPS 2004; torch 2.14.0 c10/core/TensorImpl.h;
   ExecuTorch, exir/memory_planning.py, update_tensor_lifetime and _does_not_overlap; MPI 4.1 §3.7 */
enum { MESH_INVARIANT, MESH_EXECUTION };
/* design/disjoint-pages-change-request.md, D.3.3; design/plan-to-bridge-change-request.md R4;
   design/aot-compilation-change-request.md, appendix item 7; MTL4CommandEncoder.h:32 MTL4VisibilityOptionResourceAlias;
   ExecuTorch, exir/memory_planning.py, update_tensor_lifetime; MPI 4.1 §3.7 */
/* `owned` 1: its producer's prepared operand (metal-microbench metal_ir.h Value::owned, XNNPACK's operator context held
   by its runtime from creation to destruction): its MESH_EXECUTION copies each persist across every execution */
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,lag,run,owned; };
_Static_assert(sizeof(struct mesh_value)==56 && offsetof(struct mesh_value,run)==48 &&
               offsetof(struct mesh_value,owned)==52, "mesh_value: rdma/mesh.py Value");
/* design/plan-to-bridge-change-request.md R4 */
enum { MESH_JOIN, MESH_PAGES };
/* design/plan-to-bridge-change-request.md R1; MPI 4.1 §6.1.3, MPI_TYPE_CREATE_SUBARRAY: struct mesh_strided (mesh-graph.h),
   the view's maximal contiguous
   byte ranges in order, each `visit(context, offset, length)`, the innermost dimensions contiguous with their
   elements merged into one run */
void mesh_strided_extents(const struct mesh_strided *view,void (*visit)(void *context,uint64_t offset,uint64_t length),
                          void *context);
/* `root` 0: `value` the owner function's ValueID; r + 1: the program's shared root r (the joint program's roots[r], a
   cross-call wire owned by its producing call), `value` the owner's typed ValueID of it (MESH_ABSENT: no numeric site
   on this node) */
struct mesh_declared { uint32_t source,destination,identity,kind,value,root; struct mesh_strided from,to; };
_Static_assert(sizeof(struct mesh_declared)==88 && offsetof(struct mesh_declared,from)==24 &&
               offsetof(struct mesh_declared,to)==56,"mesh_declared: rdma/mesh.py Declared");
/* design/aot-compilation-change-request.md A3; O. Sinnen and L. A. Sousa, IEEE TPDS 16(6) 2005, Definition 1 */
struct mesh_weight { double produced,consumed,read,page; };
/* design/disjoint-pages-change-request.md, Appendix C, C3.3;
   design/aot-compilation-change-request.md, appendix item 1 */
#define MESH_FROM_PRODUCED 1u
#define MESH_TO_PRODUCED 2u
#define MESH_FROM_SOURCE_READ 4u
#define MESH_TO_SOURCE_READ 8u
struct mesh_dependence { uint32_t from,to,lag,kinds; };
/* design/aot-compilation-change-request.md A8, A10 ("Its output") and appendix items 2 and 6;
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (1);
   design/torch-capture-change-request.md item 2 and "Source for item 2's execution-time estimates (2026-11-09)";
   design/algorithm-sources.md#the-plans-results-for-native-preparation */
/* design/torch-source-lowering-change-request.md item 3, "The algorithm for the order";
   design/aot-compilation-change-request.md A6; design/plan-to-bridge-change-request.md R3 and R5 */
struct mesh_task { double length; uint32_t resource,padding; };
struct mesh_function { const struct mesh_declared *declared; const struct mesh_weight *weights; const struct mesh_value *value;
                       const struct mesh_dependence *dependence; uint32_t count,values,dependences,padding;
                       uint64_t outstanding; const double *host,*work; const struct mesh_graph_dependence *edge;
                       uint32_t parts,edges; const struct mesh_task *task; const struct mesh_graph_dependence *before;
                       const uint32_t *sender,*receiver; uint32_t tasks,resources,befores,padding2;
                       struct mesh_graph_relation *relation; };
/* `relation` (design/compilation-composition-change-request.md items 3, 6.4 and 7): the compiled parts' relation (the
   Actor quotient), borrowed; edge and edges are its data.edge and data.edges; NULL for a function whose edges are a
   flat operand */
_Static_assert(sizeof(struct mesh_function)==144 && offsetof(struct mesh_function,relation)==136 &&
               offsetof(struct mesh_function,count)==32 && offsetof(struct mesh_function,outstanding)==48 &&
               offsetof(struct mesh_function,host)==56 && offsetof(struct mesh_function,parts)==80 &&
               offsetof(struct mesh_function,task)==88 && offsetof(struct mesh_function,tasks)==120 &&
               sizeof(struct mesh_task)==16,"mesh_function: rdma/mesh.py Function");
/* design/aot-compilation-change-request.md A7, A10 ("Its output" and "The calls' graph") and appendix items 2 and 6;
   design/plan-to-bridge-change-request.md R2; S. S. Bhattacharyya, S. Sriram and E. A. Lee,
   IEEE Trans. Signal Processing 45(6), 1997, §4.1, §4.3 (5), §9, Figs. 6 and 8, and Lemma 1;
   M. Lam, Software pipelining, PLDI 1988;
   L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical Memo 33;
   P. E. McKenney and J. D. Slingwine, Read-copy update, 1998;
   L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986;
   os/os_sync_wait_on_address.h:268 os_sync_wait_on_address_with_timeout; _time.h:185 clock_gettime_nsec_np */
/* design/aot-compilation-change-request.md A10 ("Its output" and "Its bound"); A. J. Bernstein, IEEE Trans. Electronic
   Computers EC-15(5), 1966; M. C. Rinard and P. C. Diniz, ACM TOPLAS 19(6), 1997;
   design/compilation-composition-change-request.md item 6, unit 6D, "6. Carry compilation results through their
   consumers" and "Complete replacement workgroup", Program relations; design/aot-compilation-change-request.md,
   appendix item 6; S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, §4.1 and §9;
   design/algorithm-sources.md#the-plans-results-for-native-preparation */
struct mesh_graph_signature;
struct mesh_graph_dependence;
struct mesh_graph_access;
enum { MESH_ORDER_FLOW, MESH_ORDER_CONFLICT, MESH_PLANNED_PROGRAM, MESH_PLANNED_SUBMIT, MESH_PLANNED_OUTSTANDING,
       MESH_PLANNED_CHAIN, MESH_PLANNED_TRANSFER, MESH_PLANNED_PART };
/* struct mesh_planned order[q]'s domain by cause[q].kind, in every result (mesh_plan, mesh_plan_local, loaded):
   MESH_PLANNED_PART call `pair`'s part vertices from 0 (its function's edges); MESH_ORDER_FLOW, MESH_ORDER_CONFLICT
   and MESH_PLANNED_CHAIN call ordinals; MESH_PLANNED_PROGRAM, _SUBMIT, _OUTSTANDING and _TRANSFER the signature's
   tasks */
struct mesh_planned_cause { uint32_t kind,pair; };
struct mesh_planned_call { uint64_t lead; uint32_t function,ordinal,every,parts,waits,padding; };
/* design/compilation-composition-change-request.md items 3, 6.4 and 7; design/plan-to-bridge-change-request.md R1, R2
   and R5; MPI 4.1 §3.7.3 and §6.1.3: original transfer t of function f's planned rows, extent[received, sending) its
   RECEIVE rows and [sending, end) its SEND rows, each mesh_graph_access one planned entry's logical bytes [start, end)
   of its demand's value (vertex: a RECEIVE row's phase field, a SEND row's 0), in the entries' scheduled order; the
   transfers of one demand share its rows, every copy the same */
struct mesh_planned_transfer { size_t received,sending,end; };
/* design/joint-plan-results.md */
struct mesh_plan_compilation;
struct mesh_plan_binding {
  struct mesh_plan_compilation *compilation;
  struct mesh_range rows,arena[2];
  uint64_t generation;
  uint64_t *plan;
  uint32_t *page;
};
void mesh_plan_binding_free(struct mesh_plan_binding *);
struct mesh_plan_geometry { uint32_t pgsz,block,rows,padding; struct mesh_range arena[2]; };
struct mesh_plan_value { struct mesh_section section; uint32_t shares; };
struct mesh_plan_actors { const double *host,*work; const struct mesh_graph_dependence *edge; uint32_t parts,edges;
                          struct mesh_graph_relation *relation; };
struct plan_schedule_result;
struct mesh_plan_function { struct mesh_plan_value *value; double *completion; uint32_t *order;
                            const struct mesh_plan_actors *actors; struct plan_schedule_result *schedule; };
_Static_assert(sizeof(struct mesh_plan_value)==32 && sizeof(struct mesh_plan_actors)==40 &&
               sizeof(struct mesh_plan_function)==40,"mesh plan results: rdma/mesh.py");
struct mesh_planned { struct mesh_planned_call *call; struct mesh_graph_signature *signature;
                      struct mesh_graph_dependence *order; uint32_t calls,padding; size_t orders;
                      struct mesh_planned_cause *cause; uint64_t horizon;
                      uint32_t functions; size_t extents,*first; struct mesh_planned_transfer *transfer;
                      struct mesh_graph_access *extent; void *owned; uint64_t used[2];
                      struct mesh_range arena[2],rows;
                      struct mesh_plan_function *result; struct mesh_plan_value *root; uint32_t results,roots; struct mesh_plan_analysis *analysis; };
/* design/planner-composition.md; design/joint-plan-results.md#archives */
_Static_assert(sizeof(struct mesh_planned_call)==32 && offsetof(struct mesh_planned_call,function)==8 &&
               offsetof(struct mesh_planned_call,waits)==24 && sizeof(struct mesh_planned)==176 &&
               offsetof(struct mesh_planned,result)==144 && offsetof(struct mesh_planned,analysis)==168,
               "mesh_planned: rdma/mesh.py Planned");
uint64_t mesh_plan_quantum(const struct hdr *);
uint64_t mesh_plan_quanta(uint64_t bytes,uint64_t quantum);
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t program,const struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
              const uint32_t *chains,uint64_t deadline,struct mesh_plan_binding *binding);

void mesh_planned_free(struct mesh_planned *);
/* design/compilation-composition-change-request.md items 3, 6.4 and 7; design/torch-capture-change-request.md item 6;
   design/aot-compilation-change-request.md A10 and appendix items 2 and 6; design/disjoint-pages-change-request.md:
   one program's lowered machine functions' own storage, no transfer. functions[f] its parts (host, work, edge: its
   compiled actors) and its values (each value's section.bytes, begin, end, kind, lag and run: run 1 one this function's
   storage holds, run 0 an interface value the program binds); call j a call of functions[calls[j]]; `order` the calls'
   retained relation, ordered[q] its edge carrying order q, `closure` its closure over the calls (row b b's
   predecessors), chains[j] call j's head (h <= j, chains[h] == h; NULL each its own); roots[r] one program root (one
   source storage: section.bytes, kind, run 1 placed, run 0 the caller's), its uses use[use_first[r] .. use_first[r +
   1]) the calls binding it to their function's run 0 value; `quantum` the alignment of every placed copy. Each function
   has `every` copies of its calls (its calls' count) per execution of the plan's horizon; copy c of function f is call
   c % every of block c / every, root copy t block t, a root's occurrences its uses (none: it meets every copy). A
   lagged, transferred or invariant value's copy meets its function's other copies as their circular lifetimes do and
   every other copy; any other two copies, of one function or not, meet unless both are of one block and every
   occurrence of one precedes every occurrence of the other (in one call by its value's begin and end, across calls by
   `closure`, never by call number). `planned` one call {lead 0, function,
   ordinal, every, parts, waits 0} per call, the horizon of their parts' Bff, its orders: each call's function edges as
   MESH_PLANNED_PART orders on that call's part vertices from 0 (cause pair the call), each order q from and to call
   ordinals (MESH_ORDER_CONFLICT where a view hazard is among its edge's causes, else MESH_ORDER_FLOW; pair q), each
   chain's successive calls (delay 0) and its last before its head (delay 1) as MESH_PLANNED_CHAIN (pair UINT32_MAX)
   on call ordinals; no signature; `bytes` the arena the caller allocates, first[] over function 0's values, ...,
   function count - 1's, then the roots, offset[first[v], first[v + 1]) each one's copies' byte offsets in it (none
   where run is 0). planned.result[f].value[j] and planned.root[r] hold the derived complete section and shares;
   section.count is its copies, shares is 1 where one copy's assigned bytes meets another copy's. The supplied
   functions and roots remain unchanged */
struct mesh_local_use { uint32_t call,value; };
_Static_assert(sizeof(struct mesh_local_use)==8,"mesh_local_use: rdma/mesh.py LocalUse");
/* compiler-emit/docs/planner-source-owner.md */
struct mesh_grouped;
struct plan_item { uint64_t element,blocks; uint32_t kind,function,index,copy,space; };
struct plan_placement {
  struct plan_item *item;
  uint64_t *count,*chain,items,elements;
  uint32_t split,table;
  struct mesh_range range[2];
  uint64_t *at,blocks;
  size_t allocations;
  uint32_t run,space;
  uint64_t used[2];
  uint64_t *size,*start,*end,*offset,*operations;
  uint32_t *function,*index,*copy;
};
struct mesh_local {
  struct mesh_planned planned;
  uint64_t bytes;
  size_t values,offsets,*first;
  uint64_t *offset;
  size_t *value_first,*demand_first;
  uint32_t *demand_of,*release_of;
  uint64_t quantum;
  struct plan_placement *placement;
};

int mesh_plan_local(const struct mesh_function *functions,uint32_t count,const uint32_t *calls,uint32_t ncalls,
                    const struct mesh_graph_relation *order,const uint32_t *ordered,uint32_t norders,
                    const struct mesh_graph_closure *closure,const uint32_t *chains,const struct mesh_value *roots,
                    uint32_t nroots,const size_t *use_first,const struct mesh_local_use *use,uint64_t quantum,
                    struct mesh_local *out);
void mesh_local_free(struct mesh_local *out);
/* design/aot-compilation-change-request.md A4 (AllGather, CollectivePermute, AllToAll) and A9;
   design/plan-to-bridge-change-request.md R2 and R5: the one canonical grouping of a function's declared rows
   declared[count] (weights[count]) into demands, on node `self` of a map of `nodes`, which mesh_plan and
   mesh_plan_program consume and their callers retain before lowering: rows of one source and identity whose from views
   are alike and whose produced and read weights are equal, their destinations distinct, are one demand, its first row
   in (source, identity, row) order its owner `first`. The caller's arrays grouped->demand[count], demand_of[count] and
   release_of[count] receive demand_of[t] row t's demand, release_of[t] the release its sends share (the demands of
   source `self`, numbered in demand order; MESH_ABSENT a row of another source) and demand[d] {source, identity, first,
   release, leaves (its destinations' set), bytes (its from view's logical bytes), pages (1: every row MESH_PAGES)};
   `transfers` count, `demands`, `releases`, `nodes` and `self` as made. EHOSTUNREACH a node past the map; EINVAL a row
   to its own source, a view out of range or of other bytes than its other, a kind past MESH_PAGES or a weight not
   positive and finite */
struct mesh_demand { uint32_t source,identity,first,release; uint64_t leaves,bytes; uint32_t pages,padding; };
struct mesh_grouped { struct mesh_demand *demand; uint32_t *demand_of,*release_of;
                      uint32_t transfers,demands,releases,nodes,self,padding; };
_Static_assert(sizeof(struct mesh_demand)==40 && offsetof(struct mesh_demand,leaves)==16 &&
               offsetof(struct mesh_demand,pages)==32 && sizeof(struct mesh_grouped)==48 &&
               offsetof(struct mesh_grouped,transfers)==24 && offsetof(struct mesh_grouped,self)==40,
               "mesh_demand, mesh_grouped: rdma/mesh.py Demand, Grouped");
int mesh_declared_group(const struct mesh_declared *declared,const struct mesh_weight *weights,uint32_t count,
                        uint32_t nodes,uint32_t self,struct mesh_grouped *grouped);
/* design/aot-compilation-change-request.md A2, A4 and A7 to A10, appendix items 2 and 6: one program's joint network
   plan, mesh_plan's over its instances functions[count] (calls[j] call j's), with mesh_plan_local's program operands
   (order, ordered, closure, chains, roots, use_first, use) and the original source family (source_calls[j] call j's
   source function of `sources`, source_outstanding[] each one's issue bound: its own rows, A10's successor and delay).
   grouped[f] function f's declared rows' grouping on this node, required (mesh_declared_group's, retained by the
   caller and borrowed by the result, out->grouped; plan_function's demand_of its view; NULL: EINVAL; one of another
   map or node: ESTALE, of another row count: EINVAL). A function's
   value of no bytes (an absent, alias or constant slot) and a run 0 value a use binds to a root (its storage the
   root's; a root 0 declared row naming it transports that root, EINVAL where its function's calls bind it to several)
   keep their ValueIDs and have no allocation: no pages, no rows, section.count 0. Its result `out` is
   mesh_plan_local's owner:
   its planned record mesh_plan's (rows, arena, used, signature), its copies rows of pages from planned.rows.first
   (first[]/offset[], value_first as mesh_plan_local's), the instance and source rows (row/order,
   source_row/source_order), and each instance's declared rows' demands (demand_of from
   demand_first[f]) with the release a sent row's demand shares (release_of; MESH_ABSENT a row this node does not
   send). Analysis retains caller-owned run 0 roots and their original views. Physical construction requires a
   transported root's assigned backing; its current run 1 placement cannot supply a run 0 root's missing binding. */
struct mesh_plan_flow;
enum { MESH_PLAN_FROM, MESH_PLAN_TO };
struct mesh_plan_endpoint { uint32_t transfer,role; const struct mesh_strided *view; };
struct mesh_plan_cuts {
  const struct mesh_plan_geometry *geometry;
  const struct mesh_plan_flow *source;
  uint32_t demand,views;
  const struct mesh_plan_endpoint *endpoint;
  uint32_t spans,padding;
  const struct mesh_graph_interval *span;
  const struct mesh_plan_cuts *previous;
  uint64_t instructions;
  uint64_t *cut,*at;
  uint32_t *route;
  uint64_t *count;
  uint64_t *previous_first,*previous_last;
  const void *mapped;
};
int mesh_plan_cut(const struct mesh_plan_geometry *,const struct mesh_plan_flow *,uint32_t,
                  uint32_t,const struct mesh_plan_endpoint *,uint32_t,const struct mesh_graph_interval *,
                  const struct mesh_plan_cuts *,struct mesh_plan_cuts **);
void mesh_plan_cut_free(struct mesh_plan_cuts *);
struct mesh_plan_demand {
  uint32_t source,identity,first;
  uint32_t mine,root;
  /* design/plan-to-bridge-change-request.md R2 */
  const struct mesh_strided *view;
  uint32_t tree,trees;
  uint32_t release;
  /* design/aot-compilation-change-request.md A9 */
  uint64_t leaves,bytes;
  uint32_t relay;
  uint64_t on;
  uint32_t forwards;
  uint32_t own;
  uint32_t set;
  uint32_t vertex;
  uint32_t value;
};
/* design/disjoint-pages-change-request.md C3.3 */
struct mesh_plan_vertex { uint32_t demand,node,write,read,reads,source; };
/* design/plan-to-bridge-change-request.md R5; M. Cowan et al., MSCCLang, ASPLOS 2023, §5.2 */
struct mesh_plan_entry { uint32_t node,direction,demand,field,communication,set; uint64_t instruction; };
/* compiler-emit/docs/physical-geometry.md */
struct mesh_plan_demand_geometry {
  uint32_t range,received,consumed,counted,sending,sends;
  uint64_t group;
};
struct mesh_plan_flow_geometry {
  size_t references;
  const struct mesh_plan_flow *source;
  const struct mesh_plan_cuts *const *cuts;
  const uint64_t *const *count;
  const uint32_t *local;
  uint64_t *lead,*comm,comms;
  struct mesh_plan_demand_geometry *demand;
  struct mesh_plan_entry *entry;
  uint64_t entries,receives,toward,group_bytes,forwarded;
  uint32_t ranges,consumed,sending;
  double *page,time[MESH_NODES_MOST];
  const void *mapped;
};
int mesh_plan_flow_geometry(const struct mesh_plan_flow *,const struct mesh_plan_cuts *const *,
                            const uint64_t *const *,const uint32_t *,struct mesh_plan_flow_geometry **);
void mesh_plan_flow_geometry_retain(struct mesh_plan_flow_geometry *);
void mesh_plan_flow_geometry_free(struct mesh_plan_flow_geometry *);
struct mesh_plan_flow {
  const struct mesh_plan_geometry *geometry;
  const struct mesh_declared *declared;
  uint32_t transfers;
  const struct mesh_value *given;
  uint32_t *value_lag;
  uint32_t givens;
  /* design/plan-to-bridge-change-request.md R2; design/disjoint-pages-change-request.md C3.3 */
  unsigned char *named;
  uint32_t *uncovered;
  struct mesh_plan_endpoint *endpoint;
  struct mesh_graph_grouping *transfer_rows;
  struct mesh_plan_cuts **cuts;
  const uint64_t **counts;
  uint32_t *cut_local;
  struct mesh_plan_flow_geometry *physical;
  uint32_t sets,*set_first,*member,*set_blocks,*set_far,*block_first,*block_set,*block_element;
  const struct mesh_weight *weights;
  const struct mesh_dependence *dependence;
  uint32_t dependences;
  struct mesh_plan_demand *demand;
  uint32_t demands,n,self,copies,start_lead;
  struct mesh_view_id view;               /* JGroups ViewId.java */
  /* of[t] row t's demand: a view of its grouping's demand_of (mesh_declared_group's; `own` the one plan_run made
     where the caller gave none, held until plan_free) */
  const uint32_t *of;
  struct mesh_grouped own;
  uint32_t pairs,from[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1],to[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  double capacity[MESH_NODES_MOST*(MESH_NODES_MOST-1)/2+1];
  size_t *tree;
  uint32_t *parent;
  uint64_t *run;
  uint32_t made;
  uint64_t *members;
  struct mesh_graph_interval *span;
  uint64_t releases;
  uint64_t blocks;
  uint64_t *child;
  /* the joint program's shared roots (mesh_plan_program), `carried[r]` where one of this plan's demands transports r;
     bound[j] the root a use binds value j to (MESH_ABSENT none, PLAN_SEVERAL several of its calls' roots), its
     storage the root's (NULL: no uses) */
  const struct mesh_value *roots;
  uint32_t nroots;
  unsigned char *carried;
  uint32_t *bound;
  struct mesh_plan_vertex *value;
  uint32_t values;
  uint32_t *read;
  uint32_t *here,local;
  uint32_t vertices;
  size_t edges;
  struct mesh_graph_dependence *edge;
  struct mesh_graph_grouping *rows;
  uint32_t *between;
  uint32_t *farthest;
  /* design/aot-compilation-change-request.md A10 */
  uint64_t outstanding,spanned,nodes;
  uint32_t calls,horizon;
  struct mesh_graph_tree **adjacency;
  uint32_t *acyclic_indegree,*acyclic_ordinal,*lag_source,*lag_target;
  struct mesh_graph_lag_graph *lag_graph;
  struct mesh_graph_lag_matrix *lags;
  struct mesh_graph_cover_result *covers;
  struct mesh_graph_touch *touch;
  uint32_t *relay;
};

struct mesh_plan_analysis {
  size_t references;
  const struct mesh_plan_geometry *geometry;
  uint64_t program;
  const struct mesh_function *function;
  uint32_t functions;
  const uint32_t *calls,*chains,*ordered,*source_calls,*orders,*causes;
  uint32_t ncalls,norders,sources,horizon;
  const uint64_t *source_outstanding;
  const struct mesh_graph_relation *relation;
  const struct mesh_graph_closure *closure;
  const struct mesh_value *root;
  uint32_t roots;
  const size_t *use_first;
  const struct mesh_local_use *use;
  const struct mesh_grouped *grouped;
  struct mesh_plan_flow *flow;
  struct mesh_graph_grouping *call_group,*source_group;
  struct mesh_graph_chain_result *chain_pairs;
  struct mesh_graph_buffer_result *bounds;
  const uint32_t *actor_function;
  uint64_t *function_lag;
  double *bound_host,*bound_work;
  struct mesh_graph_dependence *bound_edge;
  uint32_t *bound_function;
  unsigned char *carried;
  const struct mesh_nodes *nodes;
  void *owned,*environment;
};

void mesh_plan_analysis_retain(struct mesh_plan_analysis *);
void mesh_plan_analysis_free(struct mesh_plan_analysis *);
int mesh_plan_local_analyze(const struct mesh_function *,uint32_t,const uint32_t *,uint32_t,
  const struct mesh_graph_relation *,const uint32_t *,uint32_t,const struct mesh_graph_closure *,const uint32_t *,
  const struct mesh_value *,uint32_t,const size_t *,const struct mesh_local_use *,struct mesh_plan_analysis **);
int mesh_plan_program_analyze(const struct mesh_nodes *,const struct mesh_plan_geometry *,uint64_t,
  struct mesh_graph_buffer_result *,const uint32_t *,const struct mesh_function *,uint32_t,
  const struct mesh_grouped *,const uint32_t *,uint32_t,const uint32_t *,uint32_t,const uint64_t *,
  const struct mesh_graph_relation *,const uint32_t *,uint32_t,const struct mesh_graph_closure *,const uint32_t *,
  const struct mesh_value *,uint32_t,const size_t *,const struct mesh_local_use *,struct mesh_plan_analysis **);
/* routing[a*nodes->count+b] gives the symmetric capacity available to the flow solver. */
int mesh_plan_program_analyze_routed(const struct mesh_nodes *,const struct mesh_plan_geometry *,uint64_t,
  struct mesh_graph_buffer_result *,const uint32_t *,const struct mesh_function *,uint32_t,
  const struct mesh_grouped *,const uint32_t *,uint32_t,const uint32_t *,uint32_t,const uint64_t *,
  const struct mesh_graph_relation *,const uint32_t *,uint32_t,const struct mesh_graph_closure *,const uint32_t *,
  const struct mesh_value *,uint32_t,const size_t *,const struct mesh_local_use *,const double *,struct mesh_plan_analysis **);
int mesh_plan_local_complete(struct mesh_plan_analysis *,uint64_t,struct mesh_local *);
struct plan_sync { uint32_t node,function,entry,every; int64_t base; };
struct plan_entry { const struct mesh_plan_entry *source; uint32_t place; };
struct plan_span { uint32_t first,count; };
struct plan_access_query { uint32_t task,from,to;size_t dependency; };
struct plan_schedule_result {
  size_t references;
  struct mesh_plan_flow_geometry *geometry;
  const struct mesh_function *function;
  uint32_t first,tasks,instructions;
  size_t edges,entries,dependencies;
  uint32_t *reader;
  struct plan_span *consumer;
  double *w,*cost,*rate,*completion,*priority,*completed;
  uint32_t *resource,*from,*to,*link,*communication,*before,*after,*place;
  size_t *route;
  struct mesh_plan_analysis *owned;
  size_t accesses;
  const struct mesh_graph_view *access;
  struct mesh_graph_relation *access_relation;
  uint32_t *page_first,*page_edge;
  unsigned char *access_role;
  const size_t *event_first;
  const uint32_t *event_member;
  size_t *task_edge,*access_edge;
  struct mesh_graph_relation *ordering;
  uint32_t *order;
  struct mesh_graph_dependence *resource_cycle;
  size_t resource_edges,access_queries;
  struct plan_access_query *access_query;
};
/* design/flow-order-and-lowering-memorandum.md: access.key is a declared transfer, vertex a Function task;
   write selects the original from view's producer, read the to view's consumer. Supplied roles are complete,
   including whole readers; omitted roles keep sender/receiver. Event rows 3*t+[0,1,2] are the original produced,
   consumed, source-read completion memberships in Function task IDs, joined with zero-time vertices. */
int mesh_plan_schedule(struct mesh_plan_flow_geometry *,const struct mesh_function *,size_t,
                       const struct mesh_graph_view *,const size_t *,const uint32_t *,struct plan_schedule_result **);
void mesh_plan_schedule_retain(struct plan_schedule_result *);
void mesh_plan_schedule_free(struct plan_schedule_result *);
struct mesh_plan_storage_function {
  struct plan_schedule_result *schedule;
  const struct mesh_function *function;
  size_t lifetimes;
  const size_t *first,*last;
  const uint32_t *index;
  uint32_t copies,*value_lag;
  struct mesh_graph_closure before;
};
struct mesh_plan_storage {
  size_t references;
  struct mesh_plan_analysis *analysis;
  struct mesh_graph_buffer_result *bounds;
  const uint32_t *bound_function;
  uint32_t horizon;
  uint64_t *function_lag;
  struct mesh_plan_storage_function *function;
  void *mapped;
};
int mesh_plan_storage(struct mesh_plan_analysis *,struct plan_schedule_result *const *,
  struct mesh_graph_buffer_result *,const uint32_t *,const size_t *,const size_t *const *,const size_t *const *,
  const uint32_t *const *,struct mesh_plan_storage **);
int mesh_plan_storage_ordered(struct mesh_plan_analysis *,struct plan_schedule_result *const *,
  struct mesh_graph_buffer_result *,const uint32_t *,const size_t *,const size_t *const *,const size_t *const *,
  const uint32_t *const *,uint64_t *const *,struct mesh_plan_storage **);
void mesh_plan_storage_retain(struct mesh_plan_storage *);
void mesh_plan_storage_free(struct mesh_plan_storage *);
struct plan_graph_result {
  const struct mesh_plan_analysis *analysis;
  uint32_t vertices,added;
  size_t built;
  double lambda,*time;
  uint32_t *actor,*later,*from,*to,*call,*node,*observer,*key,*added_from,*added_to,*entry;
  uint64_t *lag,*lagged,*delay,*added_delay;
  size_t *first,*rows,*wait_first,*wait_order,*tell_first,*tell_order;
  struct mesh_graph_task *task;
  struct mesh_planned_cause *cause;
  struct mesh_graph_grouping *room_rows;
  struct mesh_graph_lag_graph *room_graph;
  struct mesh_graph_dependence *room_query;
  struct mesh_graph_lag_result *room_lags;
};
struct plan_step {
  const struct mesh_plan_flow *flow;
  const struct mesh_plan_flow_geometry *geometry;
  const struct mesh_function *function;
  struct mesh_plan_function *result;
  struct mesh_plan_value *root_result;
  uint32_t *item,set_item,state;
  uint64_t base;
  struct plan_entry *entry;
  uint64_t phases,counts,completions,supplied,publications,status,hops,times,words;
  uint64_t deliveries,submits,syncs,rows,room,submit_room;
  uint32_t waits,tells,*wait_row;
  struct plan_sync *sync;
  struct plan_schedule_result *schedule;
  struct plan_graph_result *graph;
  const struct mesh_plan_storage_function *storage;
};
struct plan_placement;
struct mesh_plan_compilation {
  size_t references;
  struct mesh_local result;
  struct plan_step *step;
  struct plan_placement *placement;
  uint32_t *root_item,*page,*adjacency,*toward,*place;
  size_t *rows_by,*listed;
  uint64_t places;
  void **state;
  uint64_t *state_bytes;
  void *owned;
  struct mesh_plan_storage *storage;
};
int mesh_plan_environment(struct mesh_ctx *,struct mesh_nodes *,struct mesh_plan_geometry *);
int mesh_plan_program_compile(struct mesh_plan_analysis *,struct plan_schedule_result *const *,
                              struct mesh_plan_storage *,struct mesh_plan_compilation **);
int mesh_plan_program_install(struct mesh_ctx *,uint64_t,const struct mesh_plan_compilation *,uint64_t,struct mesh_plan_binding *);
void mesh_plan_compilation_retain(struct mesh_plan_compilation *);
void mesh_plan_compilation_free(struct mesh_plan_compilation *);
int mesh_plan_compilation_save(const struct mesh_plan_compilation *,void **,size_t *);
int mesh_plan_compilation_load(struct mesh_graph_context *,const void *,size_t,void (*)(void *),void *,struct mesh_plan_compilation **);
int mesh_plan_program(struct mesh_ctx *context,uint64_t group,uint64_t program,const struct mesh_function *functions,
                      uint32_t count,const struct mesh_grouped *grouped,const uint32_t *calls,uint32_t ncalls,
                      const uint32_t *source_calls,uint32_t sources,const uint64_t *source_outstanding,
                      const struct mesh_graph_relation *order,
                      const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
                      const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
                      const struct mesh_local_use *use,uint64_t deadline,struct mesh_plan_binding *out);
/* design/compilation-composition-change-request.md items 3, 6.4 and 7: the whole solved local result, saved in one
   archive (*bytes the caller's, released with free) and loaded without planning again: each function's original value,
   declared, weight, dependence, task, before, sender and receiver operands, outstanding and compiled actors; the
   separate result sections, sharing, plan, completion and order; roots and their separate results; value_first, calls,
   chains, ordered edges, the calls' relation
   (mesh_graph_relation_save) and closure, the uses, and the planned calls, orders, causes, first and offset. `bytes`
   (8-byte aligned, EINVAL otherwise) are borrowed: the loaded Function[] descriptors and closure header are the
   result's own, every array a view of them; on success the result adopts `owner` (release(owner) once the result,
   freed by mesh_local_free, and every loaded relation are released), on failure the caller keeps it and the result is
   zeroed. `analysis` the loading compilation's one analysis context (mesh-graph.h struct mesh_graph_context; NULL
   where the result's relation holds no access relation): its relation loaded in it (mesh_graph_relation_load) */
int mesh_local_save(const struct mesh_local *local,void **bytes,size_t *length);
int mesh_local_load(struct mesh_graph_context *analysis,const void *bytes,size_t length,void (*release)(void *),
                    void *owner,struct mesh_local *local);
/* design/torch-capture-change-request.md item 6: every retained copy of a network result (local->first's values: its
   functions' values, then its roots; run 1 or 0 alike) in one mapping *view of the context's region, the view spanning
   every row of every copy (each copy section.stride rows from its first row local->offset[k], each row one block at its
   own bound page, not assumed contiguous). copy_offsets[local->offsets]: copy k's first row's byte offset in the view.
   page_offsets[local->planned.rows.count], indexed by row - local->planned.rows.first: each copy's row's page's byte
   offset in the view; defined only for those rows (not the plan's other rows, as its root row) */
void mesh_view_program(struct mesh_ctx *,const struct mesh_plan_binding *binding,uint64_t *copy_offsets,uint64_t *page_offsets,
                       struct mesh_view *view);

/* design/plan-to-bridge-change-request.md R2; design/aot-compilation-change-request.md A9 */
void mesh_host_publish(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
uint64_t mesh_host_arrived(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
uint64_t mesh_host_sent(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* design/disjoint-pages-change-request.md C3.6; design/problems-of-14b9a757-change-request.md 7.3 */
int mesh_host_wait(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution,uint32_t direction);
/* design/torch-capture-change-request.md item 6 */
void mesh_view_values(struct mesh_ctx *,const struct mesh_function *functions,const struct mesh_plan_binding *,uint32_t count,uint64_t *first,
                      uint64_t *offset,struct mesh_view *view);
#endif
