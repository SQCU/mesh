#ifndef MESH_METAL_H
#define MESH_METAL_H
#include "mesh-call.h"
/* design/prepared-machine.md#M07 */
struct mesh_metal_transport { void *memory,*publish,*consume,*stop; };
_Static_assert(sizeof(struct mesh_metal_transport)==32,"prepared Metal transport");
/* design/algorithm-sources.md#resident-metal */
int mesh_metal_transport_create(struct mesh_ctx *,void *,struct mesh_metal_transport *);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transfer_encode(struct mesh_ctx *,struct mesh_metal_transport *,void *,void *,struct mesh_section,int);
/* design/algorithm-sources.md#resident-metal */
void mesh_metal_transport_destroy(struct mesh_metal_transport *);
#endif
