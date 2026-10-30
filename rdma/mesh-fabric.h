/* design/whole-function.md#fabric: the physical fabric and a pair's write paths, as the planner (mesh-plan.h) and the
   collective's public header (nccl.h ncclMeshCommFabric, ncclMeshCommWritePaths, ncclMeshCommWriteRoutes) both name
   them.  Exact linear allocation weights: zero disables allocation to a path without deleting its capability; at
   least one weight is positive; the checked sum fits uint64_t. */
#ifndef MESH_FABRIC_H
#define MESH_FABRIC_H
#include <stdint.h>
struct mesh_cable { uint64_t wire,bandwidth; uint32_t node[2]; };
struct mesh_fabric { uint32_t nodes,cables; const struct mesh_cable *cable; };
struct mesh_paths { uint32_t count; const uint32_t *first,*cable; const uint64_t *weight; };
/* the node's GPU cores, read once a process (mesh-fabric.c): every reader's one source; 0 where none is stated */
uint32_t mesh_gpu_cores(void);
#endif
