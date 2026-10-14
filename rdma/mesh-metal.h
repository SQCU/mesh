#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* design/prepared-machine.md#M07 */
struct mesh_metal_transport { void *publication,*inputs; struct mesh_cancellation *cancel; };
struct mesh_metal_input { void *completion; uint64_t offset,stride; };
_Static_assert(sizeof(struct mesh_metal_transport)==24,"prepared Metal transport");
_Static_assert(sizeof(struct mesh_metal_input)==24,"prepared numerical input");
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_transport_create(struct mesh_ctx *,void *,struct mesh_metal_transport *);
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_receive_prepare(struct mesh_ctx *,struct mesh_metal_transport *,struct mesh_section,struct mesh_metal_input *);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
#endif
