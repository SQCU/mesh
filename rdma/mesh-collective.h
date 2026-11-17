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
struct mesh_planned_call { uint64_t plan,lead; uint32_t function,ordinal,every,parts,waits,padding; };
/* design/compilation-composition-change-request.md items 3, 6.4 and 7; design/plan-to-bridge-change-request.md R1, R2
   and R5; MPI 4.1 §3.7.3 and §6.1.3: original transfer t of function f's planned rows, extent[received, sending) its
   RECEIVE rows and [sending, end) its SEND rows, each mesh_graph_access one planned entry's logical bytes [start, end)
   of its demand's value (vertex: a RECEIVE row's phase field, a SEND row's 0), in the entries' scheduled order; the
   transfers of one demand share its rows, every copy the same */
struct mesh_planned_transfer { size_t received,sending,end; };
/* design/joint-plan-results.md */
struct mesh_plan_value { struct mesh_section section; uint32_t shares,lag; };
struct mesh_plan_actors { const double *host,*work; const struct mesh_graph_dependence *edge; uint32_t parts,edges;
                          struct mesh_graph_relation *relation; };
struct mesh_plan_function { struct mesh_plan_value *value; uint64_t plan; double *completion; uint32_t *order;
                            const struct mesh_plan_actors *actors; };
_Static_assert(sizeof(struct mesh_plan_value)==32 && sizeof(struct mesh_plan_actors)==40 &&
               sizeof(struct mesh_plan_function)==40,"mesh plan results: rdma/mesh.py");
struct mesh_planned { struct mesh_planned_call *call; struct mesh_graph_signature *signature;
                      struct mesh_graph_dependence *order; uint32_t calls,padding; size_t orders;
                      struct mesh_planned_cause *cause; uint64_t horizon;
                      uint32_t functions; size_t extents,*first; struct mesh_planned_transfer *transfer;
                      struct mesh_graph_access *extent; void *owned; uint64_t used[2];
                      struct mesh_range arena[2],rows; uint64_t generation;
                      struct mesh_plan_function *result; struct mesh_plan_value *root; uint32_t results,roots; };
/* `owned`: a loaded result's own memory and its archive's owner (mesh_plan_load, mesh_local_load), released by
   mesh_planned_free; NULL for a planned result.  `used[a]`: the blocks (m->block pages each) the plan's placement
   solved in distinct arena a, a as mesh_arena_range(m, a) (one shared arena: used[0], used[1] 0), as mesh_plan placed
   them or the loaded archive saved them; 0 for a local result.  `arena[a]`: the pages of arena a the placement's pages
   are offsets from: its reservation, {first, used[a] blocks} (mesh_plan and mesh_plan_load reserve it), equal ranges
   where one arena is shared; mesh_plan_save saves them; {0} for a
   local result.  `rows`: its reserved page-table rows, its value rows then its root row (rows.first + rows.count - 1,
   its publication's stable slot: mesh.h struct mesh_page_entry `publication`); `generation`: its publication's (0
   none), which mesh-dataflow.h mesh_retire closes */
_Static_assert(sizeof(struct mesh_planned_call)==40 && offsetof(struct mesh_planned_call,function)==16 &&
               offsetof(struct mesh_planned_call,waits)==32 && offsetof(struct mesh_planned,owned)==96 &&
               offsetof(struct mesh_planned,calls)==24 &&
               offsetof(struct mesh_planned,orders)==32 && offsetof(struct mesh_planned,cause)==40 &&
               offsetof(struct mesh_planned,horizon)==48 && offsetof(struct mesh_planned,functions)==56 &&
               offsetof(struct mesh_planned,extents)==64 && offsetof(struct mesh_planned,first)==72 &&
               offsetof(struct mesh_planned,transfer)==80 && offsetof(struct mesh_planned,extent)==88 &&
               offsetof(struct mesh_planned,used)==104 && offsetof(struct mesh_planned,arena)==120 &&
               offsetof(struct mesh_planned,rows)==136 && offsetof(struct mesh_planned,generation)==144 &&
               offsetof(struct mesh_planned,result)==152 && sizeof(struct mesh_planned)==176 &&
               sizeof(struct mesh_planned_transfer)==24 &&
               sizeof(struct mesh_planned_cause)==8,"mesh_planned: rdma/mesh.py Planned");
uint64_t mesh_plan_quantum(const struct hdr *);
uint64_t mesh_plan_quanta(uint64_t bytes,uint64_t quantum);
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t program,const struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
              const uint32_t *chains,uint64_t deadline,struct mesh_planned *planned);
int mesh_plan_save(struct mesh_ctx *,uint64_t program,const struct mesh_function *,uint32_t count,
  const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
  const uint32_t *chains,const struct mesh_planned *,void **bytes,size_t *length);
/* `analysis` the loading compilation's analysis context (mesh-graph.h struct mesh_graph_context): each function's actor
   relation loaded in it (mesh_graph_relation_load) */
int mesh_plan_load(struct mesh_ctx *,struct mesh_graph_context *analysis,uint64_t group,uint64_t program,
  const struct mesh_function *,uint32_t count,
  const uint32_t *calls,uint32_t ncalls,const uint32_t *orders,const uint32_t *causes,uint32_t norders,
  const uint32_t *chains,uint64_t deadline,const void *bytes,size_t length,void (*release)(void *),void *owner,
  struct mesh_planned *);
/* mesh_plan_load's archive (8-byte aligned `bytes`, EINVAL otherwise) restores each function's compiled parts (host,
   work, and their relation or flat edges) in result[f].actors beside the unchanged functions[f], as views of `bytes`, with
   the result's order, cause and extents; on success the result adopts `owner` (release(owner) once the result, freed
   by mesh_planned_free, and every loaded relation are released), on failure the caller keeps it.  mesh_plan and
   mesh_plan_load are each one preparation (hdr `preparing`, until `deadline`): the plan's rows and the used blocks of
   each distinct arena reserved around every live publication's (mesh_graph_greedy_by_size, pinned), its saved pages
   and rows relocated to them from the archive's origins (`arena`, `rows`), then published at its root row */
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
   `closure`, never by call number). `planned` one call {plan 0, lead 0, function,
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
/* the whole solved result's operands, as mesh_plan_local was given them (a solved result borrows them from the
   caller; a loaded one owns its Function[] descriptors and closure header and borrows every array from its archive):
   function[functions], root[roots], value_first[functions + 1] (produced here: function f's values are first[]'s
   entries value_first[f] .. value_first[f + 1], the roots from value_first[functions]), calls[ncalls], chains[ncalls]
   (NULL: each call its own head), ordered[norders] (the edges of `relation` that are orders), the calls' `closure`,
   and use_first[roots + 1] and use[]; `grouped` a network result's grouped[functions] (mesh_plan_program's operand,
   the caller's mesh_declared_group results, borrowed with their arrays: the original group records, each demand's
   first row, and the rows' demand_of and release_of; a loaded one's views of its archive), NULL a local result's;
   quantum is the original placement quantum in bytes, not arena-base alignment */
struct mesh_grouped;
struct mesh_local { struct mesh_planned planned; uint64_t bytes; size_t values,offsets; size_t *first;
                    uint64_t *offset; uint32_t functions,roots; const struct mesh_function *function;
                    const struct mesh_value *root; size_t *value_first; const uint32_t *calls,*chains,*ordered;
                    uint32_t ncalls,norders; const struct mesh_graph_relation *relation;
                    const struct mesh_graph_closure *closure; const size_t *use_first;
                    const struct mesh_local_use *use; const uint32_t *source_calls;
                    const uint64_t *source_outstanding; uint32_t sources,padding; size_t *row,*order,*source_row,
                    *source_order,*demand_first; uint32_t *demand_of,*release_of;
                    const struct mesh_grouped *grouped; uint64_t quantum; };
_Static_assert(offsetof(struct mesh_local,bytes)==176 &&
               offsetof(struct mesh_local,values)==184 && offsetof(struct mesh_local,offsets)==192 &&
               offsetof(struct mesh_local,first)==200 && offsetof(struct mesh_local,offset)==208 &&
               offsetof(struct mesh_local,functions)==216 && offsetof(struct mesh_local,function)==224 &&
               offsetof(struct mesh_local,value_first)==240 && offsetof(struct mesh_local,ncalls)==272 &&
               offsetof(struct mesh_local,relation)==280 && offsetof(struct mesh_local,use)==304 &&
               offsetof(struct mesh_local,source_calls)==312 && offsetof(struct mesh_local,sources)==328 &&
               offsetof(struct mesh_local,row)==336 && offsetof(struct mesh_local,demand_first)==368 &&
               offsetof(struct mesh_local,release_of)==384 && offsetof(struct mesh_local,grouped)==392 &&
               offsetof(struct mesh_local,quantum)==400 && sizeof(struct mesh_local)==408,
               "mesh_local: rdma/mesh.py Local");
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
   send). A shared root transported by a demand is a run 1 root (a caller's run 0 root: EINVAL; its caller stages it
   in a run 1 wire root) */
int mesh_plan_program(struct mesh_ctx *context,uint64_t group,uint64_t program,const struct mesh_function *functions,
                      uint32_t count,const struct mesh_grouped *grouped,const uint32_t *calls,uint32_t ncalls,
                      const uint32_t *source_calls,uint32_t sources,const uint64_t *source_outstanding,
                      const struct mesh_graph_relation *order,
                      const uint32_t *ordered,uint32_t norders,const struct mesh_graph_closure *closure,
                      const uint32_t *chains,const struct mesh_value *roots,uint32_t nroots,const size_t *use_first,
                      const struct mesh_local_use *use,uint64_t deadline,struct mesh_local *out);
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
/* design/compilation-composition-change-request.md items 3, 6.4 and 7; design/aot-compilation-change-request.md A8 to
   A10 and appendix item 6; design/plan-to-bridge-change-request.md R8: mesh_plan_program's whole network result `local`
   saved in one archive (*bytes the caller's, released with free): its local sections (mesh_local_save's), each
   function's declared transfers (with their strided views), weights, dependences, tasks, before, senders, receivers,
   completion and order, its plan state, the planned rows' pages, the plan table's adjacency, the planned signature and
   extents, the source family (source_calls, source_outstanding) and the instance and source rows and demands
   (row/order, source_row/source_order, demand_first, demand_of, release_of).  mesh_program_load: the same result loaded
   from `bytes` (8-byte aligned, borrowed: every array a view of them but those it relocates, its own) and published
   again without planning, one preparation as mesh_plan_load's (hdr `preparing` until `deadline`; its rows and used
   blocks reserved beside every live publication's; its pages, its value and root sections and its copies' rows
   relocated to them; its plans, rows and table written; published at its root row).  `local` is then
   mesh_plan_program's owner: mesh_local_free frees it, mesh-dataflow.h mesh_retire(planned) closes its publication.  On
   success it adopts `owner` (release(owner) once the result and every loaded relation are released); on failure the
   caller keeps it and `local` is zeroed; `analysis` as mesh_local_load's */
int mesh_program_save(struct mesh_ctx *context,uint64_t program,const struct mesh_local *local,void **bytes,
                      size_t *length);
int mesh_program_load(struct mesh_ctx *context,struct mesh_graph_context *analysis,uint64_t group,uint64_t program,
                      uint64_t deadline,const void *bytes,size_t length,void (*release)(void *),void *owner,
                      struct mesh_local *local);
/* design/torch-capture-change-request.md item 6: every retained copy of a network result (local->first's values: its
   functions' values, then its roots; run 1 or 0 alike) in one mapping *view of the context's region, the view spanning
   every row of every copy (each copy section.stride rows from its first row local->offset[k], each row one block at its
   own bound page, not assumed contiguous). copy_offsets[local->offsets]: copy k's first row's byte offset in the view.
   page_offsets[local->planned.rows.count], indexed by row - local->planned.rows.first: each copy's row's page's byte
   offset in the view; defined only for those rows (not the plan's other rows, as its root row) */
void mesh_view_program(struct mesh_ctx *,const struct mesh_local *local,uint64_t *copy_offsets,uint64_t *page_offsets,
                       struct mesh_view *view);

/* design/plan-to-bridge-change-request.md R2; design/aot-compilation-change-request.md A9 */
void mesh_host_publish(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
uint64_t mesh_host_arrived(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
uint64_t mesh_host_sent(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* design/disjoint-pages-change-request.md C3.6; design/problems-of-14b9a757-change-request.md 7.3 */
int mesh_host_wait(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution,uint32_t direction);
/* design/torch-capture-change-request.md item 6 */
void mesh_view_values(struct mesh_ctx *,const struct mesh_function *functions,const struct mesh_planned *,uint32_t count,uint64_t *first,
                      uint64_t *offset,struct mesh_view *view);
#endif
