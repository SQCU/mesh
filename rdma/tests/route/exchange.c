/* The trainer's gradient exchange as a client of the mesh (route.h): libnccl-mesh's host path over the nodes' bridges
   (rdma/NCCL.md), so the three gradients' sum crosses the Thunderbolt fabric as the mesh routes it, relays included.
   One communicator (NCCL's bootstrap at `id`, the root's host:port), one lease of the node's region for the gradients,
   and one declared program of one step: the in-place all-reduce (sum) of that lease; every later call is that step. */
#include "route.h"
#include "nccl.h"
#include <stdio.h>
#include <string.h>

static ncclComm_t comm;
static float *buf;
static size_t count;
static char failure[512];

static int fail(const char *what, ncclResult_t r) {
  snprintf(failure, sizeof failure, "%s: %s (%s)", what, ncclGetErrorString(r), comm ? ncclGetLastError(comm) : "no communicator");
  return -1;
}
int route_exchange_open(int rank, int world, const char *id, size_t n) {
  ncclUniqueId u; memset(&u, 0, sizeof u);
  if (strlen(id) >= sizeof u.internal) { snprintf(failure, sizeof failure, "the id %s is too long", id); return -1; }
  memcpy(u.internal, id, strlen(id));
  ncclResult_t r = ncclCommInitRank(&comm, world, u, rank);
  if (r != ncclSuccess) return fail("ncclCommInitRank", r);
  if ((r = ncclMemAlloc((void **)&buf, n * sizeof(float))) != ncclSuccess) return fail("ncclMemAlloc", r);
  count = n;
  if ((r = ncclMeshDeclareBegin()) != ncclSuccess) return fail("ncclMeshDeclareBegin", r);
  const ncclResult_t step = ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, comm, NULL);
  r = ncclMeshDeclareEnd();
  if (step != ncclSuccess) return fail("the declared all-reduce", step);
  if (r != ncclSuccess) return fail("ncclMeshDeclareEnd", r);
  return 0;
}
float *route_exchange_buffer(void) { return buf; }
int route_exchange_sum(void) {
  const ncclResult_t r = ncclAllReduce(buf, buf, count, ncclFloat32, ncclSum, comm, NULL);
  return r == ncclSuccess ? 0 : fail("ncclAllReduce", r);
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
