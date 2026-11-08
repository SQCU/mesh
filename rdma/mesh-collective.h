#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

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
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,lag,run,shares; uint64_t padding1; };
_Static_assert(sizeof(struct mesh_value)==64 && offsetof(struct mesh_value,run)==48 &&
               offsetof(struct mesh_value,shares)==52 && offsetof(struct mesh_value,padding1)==56,
               "mesh_value: rdma/mesh.py Value");
/* design/plan-to-bridge-change-request.md R4 */
enum { MESH_JOIN, MESH_PAGES };
/* design/plan-to-bridge-change-request.md R1;
   torch 2.14.0 c10/core/TensorImpl.h, storage_offset_ and sizes_and_strides_;
   MPI 4.1 §6.1.3, MPI_TYPE_CREATE_SUBARRAY */
struct mesh_strided { uint64_t offset,element; uint32_t dims,padding; const uint64_t *pairs; };
_Static_assert(sizeof(struct mesh_strided)==32 && offsetof(struct mesh_strided,dims)==16 &&
               offsetof(struct mesh_strided,pairs)==24,"mesh_strided: rdma/mesh.py Strided");
struct mesh_declared { uint32_t source,destination,identity,kind,value,padding; struct mesh_strided from,to; };
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
   S. S. Bhattacharyya, S. Sriram and E. A. Lee, IEEE Trans. Signal Processing 45(6), 1997, (1) */
struct mesh_function { const struct mesh_declared *declared; const struct mesh_weight *weights; struct mesh_value *value;
                       const struct mesh_dependence *dependence; uint32_t count,values,dependences,padding;
                       uint64_t outstanding,plan; };
_Static_assert(sizeof(struct mesh_function)==64 && offsetof(struct mesh_function,count)==32 &&
               offsetof(struct mesh_function,outstanding)==48 && offsetof(struct mesh_function,plan)==56,
               "mesh_function: rdma/mesh.py Function");
/* design/aot-compilation-change-request.md A7, A10 ("Its output" and "The calls' graph") and appendix items 2 and 6;
   design/plan-to-bridge-change-request.md R2; S. S. Bhattacharyya, S. Sriram and E. A. Lee,
   IEEE Trans. Signal Processing 45(6), 1997, §4.1, §4.3 (5), §9, Figs. 6 and 8, and Lemma 1;
   M. Lam, Software pipelining, PLDI 1988;
   L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, ACAPS Technical Memo 33;
   P. E. McKenney and J. D. Slingwine, Read-copy update, 1998;
   L. Lamport, On interprocess communication, Distributed Computing 1(2), 1986;
   os/os_sync_wait_on_address.h:268 os_sync_wait_on_address_with_timeout; _time.h:185 clock_gettime_nsec_np */
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t program,struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,uint64_t deadline);

/* design/plan-to-bridge-change-request.md R2; design/aot-compilation-change-request.md A9 */
void mesh_host_publish(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
uint64_t mesh_host_arrived(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
uint64_t mesh_host_sent(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* design/torch-capture-change-request.md item 6 */
void mesh_view_values(struct mesh_ctx *,const struct mesh_function *functions,uint32_t count,uint64_t *first,
                      uint64_t *offset,struct mesh_view *view);
#endif
