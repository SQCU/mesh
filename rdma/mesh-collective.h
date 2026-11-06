#ifndef MESH_COLLECTIVE_H
#define MESH_COLLECTIVE_H
#include "mesh-call.h"

/* M. Lam, Software Pipelining: An Effective Scheduling Technique for VLIW Machines, PLDI 1988, §2.3: "the minimum
   degree of unrolling, u, to implement the same schedule is simply max_i q_i."
   L. J. Hendren, G. R. Gao, E. R. Altman and C. Mukerji, A Register Allocation Framework Based on Hierarchical Cyclic
   Interval Graphs, ACAPS Technical Memo 33 (Revised), McGill, 1993, §2.2: "each variable in the original loop will be
   split into several variables depending the unrolling factor."
   ExecuTorch, exir/memory_planning.py, update_tensor_lifetime: "A tensor's lifetime are represented by the index of the
   first and last node referring that tensor in its inputs/outputs"; "Mutable buffers and device resident constants
   keep their data across calls, so they are never freed."
   MTL4CommandEncoder.h, MTL4VisibilityOptionResourceAlias: "Flushes caches to ensure that aliased virtual addresses are
   memory consistent".
   copy c's rows: section.first + c x section.stride + i, i < section.stride; execution t binds copy t mod
   section.count; lag k >= 1: execution t writes its copy after execution t - k has ended; shares = 1 iff a page bound
   in one of the value's rows is bound in another row */
enum { MESH_INVARIANT, MESH_EXECUTION };
struct mesh_value { struct mesh_section section; uint32_t kind,wire,state,begin,end,lag,run,shares; uint64_t padding1; };
_Static_assert(sizeof(struct mesh_value)==64 && offsetof(struct mesh_value,run)==48 &&
               offsetof(struct mesh_value,shares)==52 && offsetof(struct mesh_value,padding1)==56,
               "mesh_value: rdma/mesh.py Value");
/* J. Gibbons, J. Functional Programming, 1996 (MESH_PAGES) */
enum { MESH_JOIN, MESH_PAGES };
/* torch 2.14.0 c10/core/TensorImpl.h, `storage_offset_`, `sizes_and_strides_`.
   MPI 4.1 §6.1.3 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node99.htm): "The ndims parameter specifies the
   number of dimensions in the full data array and gives the number of elements in array_of_sizes, array_of_subsizes,
   and array_of_starts"; "For any dimension i, it is erroneous to specify array_of_subsizes[i] < 1".
   bytes(v) = the extents [offset + sum_d i_d x pairs[2 d + 1], + element), i_d < pairs[2 d], d < dims, the last
   dimension fastest */
struct mesh_strided { uint64_t offset,element; uint32_t dims,padding; const uint64_t *pairs; };
_Static_assert(sizeof(struct mesh_strided)==32 && offsetof(struct mesh_strided,dims)==16 &&
               offsetof(struct mesh_strided,pairs)==24,"mesh_strided: rdma/mesh.py Strided");
/* MPI 4.1 §7.1 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node114.htm): "for collective operations, the
   amount of data sent must exactly match the amount of data specified by the receiver."
   byte i of bytes(from) lands at byte i of bytes(to) */
struct mesh_declared { uint32_t source,destination,identity,kind,value,padding; struct mesh_strided from,to; };
_Static_assert(sizeof(struct mesh_declared)==88 && offsetof(struct mesh_declared,from)==24 &&
               offsetof(struct mesh_declared,to)==56,"mesh_declared: rdma/mesh.py Declared");
/* O. Sinnen, Task Scheduling for Parallel Systems, Wiley, 2007: G = (V, E, w, c) */
struct mesh_weight { double produced,consumed,read,page; };
/* event `from` of execution t happens before event `to` of execution t + lag */
#define MESH_FROM_PRODUCED 1u
#define MESH_TO_PRODUCED 2u
#define MESH_FROM_SOURCE_READ 4u
#define MESH_TO_SOURCE_READ 8u
struct mesh_dependence { uint32_t from,to,lag,kinds; };
/* MPI 4.1 §3.9, MPI_START (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node83.htm): "The request becomes
   active once the call is made. [...] The communication buffer should not be modified after the call, and until the
   operation completes."
   execution t starts at the start of execution t - lead */
struct mesh_function { const struct mesh_declared *declared; const struct mesh_weight *weights; struct mesh_value *value;
                       const struct mesh_dependence *dependence; uint32_t count,values,dependences,lead;
                       uint64_t outstanding,plan; };
_Static_assert(sizeof(struct mesh_function)==64 && offsetof(struct mesh_function,count)==32 &&
               offsetof(struct mesh_function,outstanding)==48 && offsetof(struct mesh_function,plan)==56,
               "mesh_function: rdma/mesh.py Function");
/* ETIMEDOUT only where clock_gettime_nsec_np(CLOCK_MONOTONIC) >= deadline */
int mesh_plan(struct mesh_ctx *,uint64_t group,uint64_t program,struct mesh_function *functions,uint32_t count,
              const uint32_t *calls,uint32_t ncalls,uint64_t deadline);

/* supplied := max(supplied, execution + 1) */
void mesh_host_publish(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* t = execution, c = t mod u, P its fields: 1 where every field of copy c reads mesh_phase(t / u + 1); else UINT64_MAX
   where copy c's count >= (t / u + 1) x P; else 0 */
uint64_t mesh_host_arrived(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* t = execution, c = t mod u, S its sends a copy: 0 where count_c + error_c < (t / u + 1) x S; else UINT64_MAX where
   error_c != 0; else 1 */
uint64_t mesh_host_sent(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);
/* offset[i]: value i's copy 0, copy c at offset[i] + c x section.pages x pgsz; UINT64_MAX where `run` is 0 */
int mesh_view_values(struct mesh_ctx *,const struct mesh_function *functions,uint32_t count,uint64_t *offset,
                     struct mesh_view *view);
#endif
