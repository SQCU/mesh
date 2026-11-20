/* The trainer's gradient exchange as a client of the mesh (route.h): libnccl-mesh's Metal path over the nodes' bridges
   (rdma/NCCL.md), so the gradients' sum crosses the Thunderbolt fabric as the mesh routes it, relays included, its
   combining and its waits GPU kernels in the caller's command buffer (the host path spins a CPU core on its landings
   and combines on the CPU). One communicator (NCCL's bootstrap at `id`, the root's host:port), one lease of the node's
   region for the gradients (its Metal buffer the lease's own), and one declared program of one step: the in-place
   all-reduce (sum) of that lease on a Metal stream; every later call is that step. */
#include "route.h"
#include "nccl.h"
#include <stdio.h>
#include <string.h>

static ncclComm_t comm;
static float *buf;
static size_t count;
static ncclMeshBuffer lease;   /* the lease's Metal buffer and offset */
static char failure[512];

static int fail(const char *what, ncclResult_t r) {
  snprintf(failure, sizeof failure, "%s: %s (%s)", what, ncclGetErrorString(r), comm ? ncclGetLastError(comm) : "no communicator");
  return -1;
}
int route_exchange_open(int rank, int world, const char *id, size_t n, void *commandBuffer) {
  ncclUniqueId u; memset(&u, 0, sizeof u);
  if (strlen(id) >= sizeof u.internal) { snprintf(failure, sizeof failure, "the id %s is too long", id); return -1; }
  memcpy(u.internal, id, strlen(id));
  ncclResult_t r = ncclCommInitRank(&comm, world, u, rank);
  if (r != ncclSuccess) return fail("ncclCommInitRank", r);
  if ((r = ncclMemAlloc((void **)&buf, n * sizeof(float))) != ncclSuccess) return fail("ncclMemAlloc", r);
  count = n;
  if ((r = ncclMeshMemBuffer(buf, &lease.buffer, &lease.offset)) != ncclSuccess) return fail("ncclMeshMemBuffer", r);
  ncclMeshStream st; memset(&st, 0, sizeof st); st.commandBuffer = commandBuffer;   /* a declaration's: its buffer not used */
  if ((r = ncclMeshDeclareBegin()) != ncclSuccess) return fail("ncclMeshDeclareBegin", r);
  const ncclResult_t step = ncclAllReduce(&lease, &lease, count, ncclFloat32, ncclSum, comm, (cudaStream_t)&st);
  r = ncclMeshDeclareEnd();
  if (step != ncclSuccess) return fail("the declared all-reduce", step);
  if (r != ncclSuccess) return fail("ncclMeshDeclareEnd", r);
  return 0;
}
float *route_exchange_buffer(void) { return buf; }
void *route_exchange_metal(size_t *offset) { *offset = lease.offset; return lease.buffer; }
/* the step encoded into `commandBuffer` (the caller commits it and waits for it, then calls route_exchange_status) */
int route_exchange_encode(void *commandBuffer) {
  ncclMeshStream st; memset(&st, 0, sizeof st); st.commandBuffer = commandBuffer;
  const ncclResult_t r = ncclAllReduce(&lease, &lease, count, ncclFloat32, ncclSum, comm, (cudaStream_t)&st);
  return r == ncclSuccess ? 0 : fail("ncclAllReduce", r);
}
int route_exchange_status(void) {
  ncclResult_t a = ncclSuccess;
  const ncclResult_t r = ncclCommGetAsyncError(comm, &a);
  return r == ncclSuccess && a == ncclSuccess ? 0 : fail("the step's command buffer", r != ncclSuccess ? r : a);
}
const char *route_exchange_failure(void) { return failure; }
/* the communicator destroyed (its program ended: one step a period, so always after a whole number of executions) and
   the lease freed, so no program of a client gone with pieces outstanding is left on the bridges (one such restarts its
   order at the next declaration, losing every program there: design/declared-programs.md) */
void route_exchange_close(void) {
  if (!comm) return;
  ncclCommDestroy(comm); comm = 0;
  if (buf) { ncclMemFree(buf); buf = 0; }
}
