#ifndef MESH_CALL_H
#define MESH_CALL_H
#include "mesh-dataflow.h"
#include "mesh-section.h"

/* S. S. Bhattacharyya, S. Sriram and E. A. Lee, Optimizing synchronization in multiprocessor DSP systems, IEEE Trans.
   Signal Processing 45(6), 1997, (7): "start(vi, k) ≥ end(vj, k − delay((vj, vi)))".
   L. Lamport, Time, clocks, and the ordering of events in a distributed system, CACM 21(7), 1978, IR1: "Each process
   Pi increments Ci between any two successive events".
   X. Défago, A. Schiper and P. Urbán, Total order broadcast and multicast algorithms: taxonomy and survey, ACM Comput.
   Surv. 36(4), 2004, §7.1: "a sender sends m to the sequencer".
   JGroups ViewId (https://raw.githubusercontent.com/belaban/JGroups/master/src/org/jgroups/ViewId.java): "ViewIds are
   used for ordering views (each view has a ViewId and a list of members)." */
int mesh_transfers_start(struct mesh_ctx *,uint64_t plan,uint64_t executions);
/* MPI 4.1 §3.9 (https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node83.htm), MPI_START: "The request becomes
   active once the call is made."
   NCCL User Guide, CUDA Stream Semantics (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html):
   "The NCCL call returns when the operation has been effectively enqueued to the given stream, or returns an error.
   The collective operation is then executed asynchronously on the CUDA device." */
void mesh_host_started(struct mesh_ctx *,uint64_t plan,uint32_t transfer,uint64_t execution);

#endif
